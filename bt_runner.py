"""
回测运行器 - 统一管理数据加载、回测执行、结果分析
"""
import backtrader as bt
import datetime
import os
import pandas as pd
from typing import Optional, Dict, Any, List
from bt_base import StrategyAtom, TradeRecorder


class Runner:
    """
    回测运行器
    
    用法:
        runner = Runner('data.csv', timeframe='d1')
        result = runner.run(MyAtom())
    """
    
    # 默认配置
    DEFAULT_CASH = 100000
    DEFAULT_COMMISSION = 0.001
    
    # K线周期映射
    TIMEFRAME_MAP = {
        'm1': (bt.TimeFrame.Minutes, 1),
        'm5': (bt.TimeFrame.Minutes, 5),
        'm15': (bt.TimeFrame.Minutes, 15),
        'm30': (bt.TimeFrame.Minutes, 30),
        'h1': (bt.TimeFrame.Minutes, 60),
        'h4': (bt.TimeFrame.Minutes, 240),
        'd1': (bt.TimeFrame.Days, 1),
    }
    
    # SharpeRatio统计周期配置（根据K线周期自动选择合适的统计粒度）
    SHARPE_TIMEFRAME_MAP = {
        'm1': (bt.TimeFrame.Minutes, 60),     # 1分钟K线 -> 按小时统计收益
        'm5': (bt.TimeFrame.Minutes, 60),     # 5分钟K线 -> 按小时统计收益
        'm15': (bt.TimeFrame.Minutes, 240),   # 15分钟K线 -> 按4小时统计收益
        'm30': (bt.TimeFrame.Minutes, 240),   # 30分钟K线 -> 按4小时统计收益
        'h1': (bt.TimeFrame.Days, 1),         # 1小时K线 -> 按天统计收益
        'h4': (bt.TimeFrame.Days, 1),         # 4小时K线 -> 按天统计收益
        'd1': (bt.TimeFrame.Weeks, 1),        # 日线 -> 按周统计收益
    }
    
    # 基础分析器（不包含SharpeRatio，由run()动态添加）
    BASE_ANALYZERS = [
        (bt.analyzers.DrawDown, {}),
        (bt.analyzers.Returns, {}),
        (bt.analyzers.TradeAnalyzer, {}),
        (TradeRecorder, {}),
    ]
    
    def __init__(
        self,
        data_path: str,
        timeframe: str = 'd1',
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        cash: float = DEFAULT_CASH,
        commission: float = DEFAULT_COMMISSION,
    ):
        self.data_path = data_path
        self.timeframe = timeframe.lower()
        self.start_date = start_date or datetime.datetime(1900, 1, 1)
        self.end_date = end_date or datetime.datetime.now()
        self.cash = cash
        self.commission = commission
        
        # 验证timeframe
        if self.timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"不支持的周期: {timeframe}, 可选: {list(self.TIMEFRAME_MAP.keys())}")
    
    def run(self, atom: StrategyAtom, save_trades: bool = True, plot: bool = True) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            atom: 策略原子实例
            save_trades: 是否保存交易记录
            plot: 是否生成图表
        
        Returns:
            包含回测结果的字典
        """
        cerebro = bt.Cerebro()
        
        # 加载数据
        self._load_data(cerebro)
        
        # 配置broker
        cerebro.broker.setcash(self.cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # 添加基础分析器
        for analyzer_cls, kwargs in self.BASE_ANALYZERS:
            cerebro.addanalyzer(analyzer_cls, **kwargs)
        
        # 根据timeframe动态添加SharpeRatio分析器
        sharpe_tf, sharpe_comp = self.SHARPE_TIMEFRAME_MAP[self.timeframe]
        cerebro.addanalyzer(
            bt.analyzers.SharpeRatio,
            riskfreerate=0.0,
            annualize=True,  # 开启年化
            timeframe=sharpe_tf,
            compression=sharpe_comp,
            _name='sharperatio'
        )
        
        # 从Atom获取策略（不传递params，由Atom内部处理）
        strategy_cls = atom.strategy_cls()
        cerebro.addstrategy(strategy_cls)
        
        # 从Atom获取Sizer
        sizer_cls = atom.sizer_cls()
        if sizer_cls:
            cerebro.addsizer(sizer_cls)
        else:
            cerebro.addsizer(bt.sizers.FixedSize, stake=1)
        
        # 运行
        sharpe_tf, sharpe_comp = self.SHARPE_TIMEFRAME_MAP[self.timeframe]
        sharpe_period_desc = self._get_sharpe_period_description(sharpe_tf, sharpe_comp)
        
        print(f'\n运行策略: {atom.name}')
        print(f'时间周期: {self.timeframe}')
        print(f'夏普统计周期: {sharpe_period_desc}')
        print(f'初始资金: ${self.cash:,.0f}')
        print('-' * 40)
        
        results = cerebro.run()
        strat = results[0]
        
        # 收集结果
        result = self._collect_results(cerebro, strat, atom)
        
        # 保存交易记录
        if save_trades:
            self._save_trades(strat, atom)
        
        # 生成图表
        if plot:
            self._plot(cerebro)
        
        return result
    
    def run_multiple(self, atoms: List[StrategyAtom], save_trades: bool = False, plot: bool = False) -> List[Dict[str, Any]]:
        """
        运行多个策略对比
        """
        results = []
        for atom in atoms:
            result = self.run(atom, save_trades=save_trades, plot=plot)
            results.append(result)
        
        # 打印对比表
        print('\n' + '=' * 60)
        print('策略对比')
        print('=' * 60)
        print(f"{'策略名称':<20} {'收益率':>10} {'夏普':>8} {'回撤':>8} {'胜率':>8}")
        print('-' * 60)
        for r in results:
            print(f"{r['name']:<20} {r['return_pct']:>9.2f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% {r['win_rate']:>7.2f}%")
        
        return results
    
    def _load_data(self, cerebro: bt.Cerebro):
        """加载数据"""
        tf, compression = self.TIMEFRAME_MAP[self.timeframe]
        
        # 原始分钟数据
        data = bt.feeds.GenericCSVData(
            dataname=self.data_path,
            fromdate=self.start_date,
            todate=self.end_date,
            dtformat='%Y-%m-%d %H:%M:%S',
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            timeframe=bt.TimeFrame.Minutes,
            compression=1
        )
        
        # 重采样
        if self.timeframe == 'm1':
            cerebro.adddata(data)
        else:
            cerebro.resampledata(data, timeframe=tf, compression=compression)
    
    def _get_sharpe_period_description(self, timeframe, compression) -> str:
        """生成Sharpe统计周期的人类可读描述"""
        if timeframe == bt.TimeFrame.Minutes:
            if compression == 60:
                return '按小时统计 (1h)'
            elif compression == 240:
                return '按4小时统计 (4h)'
            else:
                return f'按{compression}分钟统计'
        elif timeframe == bt.TimeFrame.Days:
            return '按天统计 (1d)'
        elif timeframe == bt.TimeFrame.Weeks:
            return '按周统计 (1w)'
        elif timeframe == bt.TimeFrame.Months:
            return '按月统计 (1m)'
        else:
            return f'timeframe={timeframe}, compression={compression}'
    
    def _collect_results(self, cerebro: bt.Cerebro, strat, atom: StrategyAtom) -> Dict[str, Any]:
        """收集回测结果"""
        final_value = cerebro.broker.getvalue()
        return_pct = (final_value / self.cash - 1) * 100
        
        # 夏普比率
        sharpe = strat.analyzers.sharperatio.get_analysis()
        sharpe_ratio = sharpe.get('sharperatio')
        
        # 标记是否使用了原生计算
        used_native_sharpe = (sharpe_ratio is not None and sharpe_ratio != 0)
        
        # 如果sharperatio为None，尝试手动计算
        if sharpe_ratio is None:
            try:
                import numpy as np
                # 方法1: 从账户价值历史计算
                if hasattr(strat, '_valuelist') and len(strat._valuelist) > 1:
                    values = np.array(strat._valuelist)
                    returns = np.diff(values) / values[:-1]  # 计算收益率
                    if len(returns) > 0 and returns.std() > 0:
                        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 年化
                        print(f'  [计算方法: 账户价值序列, 样本数={len(returns)}]')
                    else:
                        sharpe_ratio = 0.0
                else:
                    # 方法2: 从Returns分析器获取
                    returns_analyzer = strat.analyzers.returns.get_analysis()
                    # 尝试多个可能的返回值结构
                    returns_list = []
                    if hasattr(returns_analyzer, 'get') and 'rtot' in returns_analyzer:
                        rtot = returns_analyzer['rtot']
                        if isinstance(rtot, dict):
                            returns_list = list(rtot.values())
                        elif isinstance(rtot, (list, tuple)):
                            returns_list = list(rtot)
                    
                    if len(returns_list) > 1:
                        returns_array = np.array(returns_list)
                        if returns_array.std() > 0:
                            sharpe_ratio = returns_array.mean() / returns_array.std()
                            print(f'  [计算方法: Returns分析器, 样本数={len(returns_list)}]')
                        else:
                            sharpe_ratio = 0.0
                    else:
                        # 方法3: 从交易记录计算
                        recorder = strat.analyzers.traderecorder.get_analysis()
                        trades = recorder.get('trades', [])
                        if len(trades) > 1:
                            pnls = np.array([t['pnlcomm'] for t in trades])
                            if pnls.std() > 0:
                                sharpe_ratio = pnls.mean() / pnls.std()
                                print(f'  [计算方法: 交易PnL序列, 样本数={len(trades)}]')
                            else:
                                sharpe_ratio = 0.0
                        else:
                            sharpe_ratio = 0.0
                            print(f'  [警告: 无足够数据计算夏普比率 - 交易次数={len(trades)}]')
            except Exception as e:
                sharpe_ratio = 0.0
                print(f'  [夏普比率计算失败: {e}]')
        
        if sharpe_ratio is None:
            sharpe_ratio = 0.0
        
        # 回撤
        dd = strat.analyzers.drawdown.get_analysis()
        max_dd = dd['max']['drawdown']
        
        # 交易统计
        recorder = strat.analyzers.traderecorder.get_analysis()
        
        result = {
            'name': atom.name,
            'final_value': final_value,
            'return_pct': return_pct,
            'sharpe': sharpe_ratio,
            'max_dd': max_dd,
            'total_trades': recorder['total'],
            'win_rate': recorder.get('win_rate', 0),
            'total_pnl': recorder.get('total_pnl', 0),
        }
        
        # 打印结果
        sharpe_tf, sharpe_comp = self.SHARPE_TIMEFRAME_MAP[self.timeframe]
        sharpe_desc = self._get_sharpe_period_description(sharpe_tf, sharpe_comp)
        
        print(f'\n回测结果:')
        print(f'  最终资金: ${final_value:,.2f}')
        print(f'  收益率: {return_pct:.2f}%')
        if used_native_sharpe:
            print(f'  夏普比率: {sharpe_ratio:.2f} ✓原生计算({sharpe_desc})')
        else:
            print(f'  夏普比率: {sharpe_ratio:.2f}')
        print(f'  最大回撤: {max_dd:.2f}%')
        print(f'  交易次数: {recorder["total"]}')
        print(f'  胜率: {recorder.get("win_rate", 0):.2f}%')
        
        return result
    
    def _save_trades(self, strat, atom: StrategyAtom):
        """保存交易记录"""
        records = strat.get_trade_records() if hasattr(strat, 'get_trade_records') else []
        if not records:
            return
        
        output_dir = 'backtest_results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{output_dir}/trades_{atom.name}_{self.timeframe}_{timestamp}.csv'
        
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)
        print(f'\n交易记录已保存: {filename}')
    
    def _plot(self, cerebro: bt.Cerebro):
        """生成图表"""
        try:
            from plot_utils import plot_backtest_results
            plot_backtest_results(cerebro, timeframe=self.timeframe)
        except Exception as e:
            print(f'绘图失败: {e}')

