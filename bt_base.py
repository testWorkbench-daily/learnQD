"""
策略原子基类和基础组件
"""
import backtrader as bt
from abc import ABC, abstractmethod
from typing import Type, Optional, Dict, Any, List


class StrategyAtom(ABC):
    """
    策略原子基类
    
    继承此类来实现一个完整的交易策略，包括：
    - 策略逻辑 (Strategy)
    - 仓位管理 (Sizer)
    - 自定义指标 (Indicators)
    """
    
    name: str = "unnamed"
    params: Dict[str, Any] = {}
    
    @abstractmethod
    def strategy_cls(self) -> Type[bt.Strategy]:
        """返回策略类"""
        pass
    
    def sizer_cls(self) -> Optional[Type[bt.Sizer]]:
        """返回Sizer类，默认None使用系统默认"""
        return None
    
    def indicators(self) -> List[Type[bt.Indicator]]:
        """返回自定义指标类列表"""
        return []
    
    def analyzers(self) -> List[Type[bt.Analyzer]]:
        """返回自定义分析器类列表"""
        return []


# =============================================================================
# 基础Sizer
# =============================================================================
class PercentSizer(bt.Sizer):
    """按账户百分比下单"""
    params = (('pct', 0.1),)
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return int(cash * self.p.pct / data.close[0])
        return self.broker.getposition(data).size


class FixedRiskSizer(bt.Sizer):
    """固定风险仓位管理"""
    params = (
        ('risk_pct', 0.02),
        ('stop_pct', 0.05),
    )
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            risk_amount = self.broker.getvalue() * self.p.risk_pct
            stop_distance = data.close[0] * self.p.stop_pct
            return max(int(risk_amount / stop_distance), 1)
        return self.broker.getposition(data).size


# =============================================================================
# 基础Analyzer
# =============================================================================
class TradeRecorder(bt.Analyzer):
    """交易记录器"""
    
    def __init__(self):
        self.trades = []
        self.trade_count = 0
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            self.trades.append({
                'id': self.trade_count,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'barlen': trade.barlen,
                'size': trade.size,
                'price': trade.price,
            })
    
    def get_analysis(self):
        if not self.trades:
            return {'total': 0, 'trades': []}
        
        wins = [t for t in self.trades if t['pnlcomm'] > 0]
        losses = [t for t in self.trades if t['pnlcomm'] <= 0]
        
        return {
            'total': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_pnl': sum(t['pnlcomm'] for t in self.trades),
            'avg_pnl': sum(t['pnlcomm'] for t in self.trades) / len(self.trades),
            'trades': self.trades,
        }


class DailyValueRecorder(bt.Analyzer):
    """记录每日组合价值和收益率"""
    
    def __init__(self):
        self.daily_values = []
        self.prev_value = None
        self.initial_value = None
    
    def start(self):
        # 记录初始资金
        self.initial_value = self.strategy.broker.getvalue()
        self.prev_value = self.initial_value
    
    def next(self):
        # 每个bar记录一次价值(后续会根据timeframe过滤)
        current_value = self.strategy.broker.getvalue()
        dt = self.strategy.datetime.datetime(0)
        
        daily_return = 0.0
        if self.prev_value > 0:
            daily_return = (current_value - self.prev_value) / self.prev_value
        
        self.daily_values.append({
            'datetime': dt,
            'portfolio_value': current_value,
            'daily_return': daily_return,
            'cumulative_return': (current_value - self.initial_value) / self.initial_value
        })
        
        self.prev_value = current_value
    
    def get_analysis(self):
        return {
            'daily_values': self.daily_values,
            'initial_value': self.initial_value,
            'final_value': self.prev_value if self.prev_value else self.initial_value
        }


# =============================================================================
# 基础Strategy (带交易记录功能)
# =============================================================================
class BaseStrategy(bt.Strategy):
    """
    基础策略类，提供通用功能：
    - 交易记录
    - 日志输出
    - 常用指标
    """
    params = (
        ('printlog', False),
    )
    
    def __init__(self):
        self.order = None
        self.trade_records = []
        self.trade_count = 0
        # 持仓成本追踪
        self.position_cost = 0.0      # 总成本
        self.position_size = 0        # 总持仓数量
        self.position_comm = 0.0      # 累计手续费
    
    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt}] {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status == order.Completed:
            dt = self.datas[0].datetime.datetime(0)
            exec_price = order.executed.price
            exec_size = abs(order.executed.size)
            exec_comm = order.executed.comm
            
            if order.isbuy():
                # 买入：累加成本
                self.position_cost += exec_price * exec_size
                self.position_size += exec_size
                self.position_comm += exec_comm
                self._record_trade(dt, 'BUY', order, pnl=0.0)
            else:
                # 卖出：计算盈亏
                pnl = 0.0
                if self.position_size > 0:
                    # 计算平均成本
                    avg_cost = self.position_cost / self.position_size
                    # 计算本次卖出的盈亏（不含手续费）
                    pnl = (exec_price - avg_cost) * exec_size
                    # 减去手续费（买入时的分摊 + 卖出时的）
                    comm_per_unit = self.position_comm / self.position_size
                    pnl -= (comm_per_unit * exec_size + exec_comm)
                    
                    # 更新持仓（按比例减少成本和手续费）
                    if exec_size >= self.position_size:
                        # 全部平仓
                        self.position_cost = 0.0
                        self.position_size = 0
                        self.position_comm = 0.0
                    else:
                        # 部分平仓
                        ratio = exec_size / self.position_size
                        self.position_cost -= self.position_cost * ratio
                        self.position_comm -= self.position_comm * ratio
                        self.position_size -= exec_size
                
                self._record_trade(dt, 'SELL', order, pnl=pnl)
        
        self.order = None
    
    def _record_trade(self, dt, trade_type, order, pnl=0.0):
        self.trade_count += 1
        self.trade_records.append({
            'trade_id': self.trade_count,
            'datetime': dt,
            'type': trade_type,
            'price': order.executed.price,
            'size': order.executed.size,
            'value': order.executed.value,
            'commission': order.executed.comm,
            'portfolio_value': self.broker.getvalue(),
            'cash': self.broker.getcash(),
            'pnl': pnl,
        })
    
    def get_trade_records(self):
        return self.trade_records

