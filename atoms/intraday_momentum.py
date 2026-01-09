"""
日内动量策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer
import datetime


class IntradayMomentumAtom(StrategyAtom):
    """
    日内动量策略
    
    上午强势的品种下午倾向于继续强势。
    利用日内动量延续性（Intraday Momentum Continuation）。
    
    核心原理：
    - 动量延续效应在日内显著
    - 上午涨幅大的品种下午继续上涨概率高
    - 上午跌幅大的品种下午继续下跌概率高
    
    信号规则：
    - 观察期：开盘至11:30（上午收盘）
    - 交易期：13:00至14:55（下午交易）
    - 做多：上午涨幅大于阈值
    - 做空：上午跌幅大于阈值
    - 尾盘平仓：14:55之后必须平仓
    
    风控：
    - 单笔风险控制
    - 尾盘强制平仓
    - 止损保护
    
    特点：
    - 纯日内策略，不隔夜
    - 需要分钟级数据
    - 适合股票、股指、商品期货
    - 简单有效，易于执行
    """
    
    name = "intraday_momentum"
    params = {'momentum_threshold': 0.5, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        momentum_threshold = self.params['momentum_threshold']
        
        class Strategy(BaseStrategy):
            params = (
                ('momentum_threshold', momentum_threshold),  # 动量阈值（百分比）
            )
            
            def __init__(self):
                super().__init__()
                # 记录当前日期
                self.current_date = None
                # 上午开盘价和收盘价
                self.morning_open = None
                self.morning_close = None
                # 上午收益率
                self.morning_return = None
                # 是否已在观察期
                self.in_observation = False
                # 是否已在交易期
                self.in_trading = False
                # 是否已交易
                self.traded_today = False
            
            def next(self):
                if self.order:
                    return
                
                # 获取当前时间
                current_dt = self.data.datetime.datetime(0)
                today = current_dt.date()
                current_time = current_dt.time()
                
                # 检查是否是新的一天
                if self.current_date != today:
                    # 新的一天，重置所有变量
                    self.current_date = today
                    self.morning_open = None
                    self.morning_close = None
                    self.morning_return = None
                    self.in_observation = False
                    self.in_trading = False
                    self.traded_today = False
                
                # 定义时间段（中国市场）
                morning_start = datetime.time(9, 30)   # 上午开盘
                morning_end = datetime.time(11, 30)    # 上午收盘
                afternoon_start = datetime.time(13, 0) # 下午开盘
                afternoon_end = datetime.time(14, 55)  # 下午收盘前
                market_close = datetime.time(15, 0)    # 收盘
                
                # 观察期：记录上午开盘价
                if morning_start <= current_time < morning_end:
                    if not self.in_observation:
                        self.in_observation = True
                        # 记录上午开盘价
                        if self.morning_open is None:
                            self.morning_open = self.data.open[0]
                    
                    # 更新上午收盘价（最后一个bar的收盘价）
                    self.morning_close = self.data.close[0]
                
                # 上午收盘后计算收益率
                elif morning_end <= current_time < afternoon_start:
                    if self.in_observation and self.morning_return is None:
                        # 计算上午收益率
                        if self.morning_open and self.morning_close:
                            self.morning_return = (self.morning_close - self.morning_open) / self.morning_open * 100
                
                # 交易期：下午开盘后根据上午收益率交易
                elif afternoon_start <= current_time < afternoon_end:
                    if not self.in_trading:
                        self.in_trading = True
                    
                    # 如果还没有交易过且有上午收益率
                    if not self.traded_today and self.morning_return is not None:
                        current_price = self.data.close[0]
                        
                        if not self.position:
                            # 做多信号：上午涨幅大于阈值
                            if self.morning_return > self.p.momentum_threshold:
                                self.order = self.buy()
                                self.traded_today = True
                            
                            # 做空信号：上午跌幅大于阈值（收益率为负）
                            elif self.morning_return < -self.p.momentum_threshold:
                                self.order = self.sell()
                                self.traded_today = True
                
                # 尾盘平仓：14:55之后必须平仓
                elif current_time >= afternoon_end:
                    if self.position:
                        self.order = self.close()
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class IntradayMomentum_0_5(IntradayMomentumAtom):
    """日内动量策略：0.5%阈值（标准）"""
    name = "intraday_mom_0_5"
    params = {'momentum_threshold': 0.5, 'risk_pct': 0.1}


class IntradayMomentum_1_0(IntradayMomentumAtom):
    """日内动量策略：1.0%阈值（中等）"""
    name = "intraday_mom_1_0"
    params = {'momentum_threshold': 1.0, 'risk_pct': 0.1}


class IntradayMomentum_1_5(IntradayMomentumAtom):
    """日内动量策略：1.5%阈值（较高）"""
    name = "intraday_mom_1_5"
    params = {'momentum_threshold': 1.5, 'risk_pct': 0.1}


class IntradayMomentum_2_0(IntradayMomentumAtom):
    """日内动量策略：2.0%阈值（高）"""
    name = "intraday_mom_2_0"
    params = {'momentum_threshold': 2.0, 'risk_pct': 0.1}


class IntradayMomentum_0_3(IntradayMomentumAtom):
    """日内动量策略：0.3%阈值（敏感）"""
    name = "intraday_mom_0_3"
    params = {'momentum_threshold': 0.3, 'risk_pct': 0.1}


class IntradayMomentum_Aggressive(IntradayMomentumAtom):
    """激进日内动量：0.3%阈值"""
    name = "intraday_mom_aggressive"
    params = {'momentum_threshold': 0.3, 'risk_pct': 0.1}


class IntradayMomentum_Conservative(IntradayMomentumAtom):
    """保守日内动量：2.0%阈值"""
    name = "intraday_mom_conservative"
    params = {'momentum_threshold': 2.0, 'risk_pct': 0.1}


class IntradayMomentum_Moderate(IntradayMomentumAtom):
    """适度日内动量：1.0%阈值"""
    name = "intraday_mom_moderate"
    params = {'momentum_threshold': 1.0, 'risk_pct': 0.1}
