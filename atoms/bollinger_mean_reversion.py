"""
布林带回归策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer


class BollingerMeanReversionAtom(StrategyAtom):
    """
    布林带回归策略（Bollinger Bands Mean Reversion）
    
    核心逻辑：
    价格触及布林带上下轨后倾向于回归中轨。适用于震荡市，趋势市慎用。
    
    布林带计算：
    - 中轨 = MA(period)
    - 上轨 = 中轨 + devfactor × 标准差
    - 下轨 = 中轨 - devfactor × 标准差
    - 带宽 = (上轨 - 下轨) / 中轨
    
    信号规则：
    - 做多：价格触及或跌破下轨
    - 做空：价格触及或突破上轨
    - 平仓：价格回归中轨
    - 止损：价格继续远离，超过3倍标准差止损
    
    趋势过滤：
    - 只在带宽收窄时做回归（低波动期）
    - 带宽扩张时暂停该策略
    """
    
    name = "bollinger_mean_reversion"
    params = {'period': 20, 'devfactor': 2.0, 'bandwidth_threshold': 0.05, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        period = self.params['period']
        devfactor = self.params['devfactor']
        bandwidth_threshold = self.params['bandwidth_threshold']
        
        class Strategy(BaseStrategy):
            params = (
                ('period', period),
                ('devfactor', devfactor),
                ('bandwidth_threshold', bandwidth_threshold),
            )
            
            def __init__(self):
                super().__init__()
                # 布林带指标
                self.boll = bt.ind.BollingerBands(
                    period=self.p.period,
                    devfactor=self.p.devfactor
                )
                
                # 中轨、上轨、下轨
                self.mid = self.boll.mid
                self.top = self.boll.top
                self.bot = self.boll.bot
                
                # 计算带宽（用于过滤）
                self.bandwidth = (self.top - self.bot) / self.mid
                
                # 记录进场价格
                self.entry_price = None
            
            def next(self):
                if self.order:
                    return
                
                # 当前价格和布林带值
                price = self.data.close[0]
                mid_line = self.mid[0]
                top_line = self.top[0]
                bot_line = self.bot[0]
                bw = self.bandwidth[0]
                
                # 计算标准差（用于止损）
                std_dev = (top_line - mid_line) / self.p.devfactor
                
                if not self.position:
                    # 趋势过滤：只在低波动期（带宽收窄）交易
                    if bw < self.p.bandwidth_threshold:
                        # 做多信号：价格触及或跌破下轨
                        if price <= bot_line:
                            self.order = self.buy()
                            self.entry_price = price
                        
                        # 做空信号：价格触及或突破上轨
                        elif price >= top_line:
                            self.order = self.sell()
                            self.entry_price = price
                
                else:
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. 获利：价格回归中轨
                        if price >= mid_line:
                            self.order = self.close()
                            self.entry_price = None
                        
                        # 2. 止损：价格继续下跌，超过3倍标准差
                        elif self.entry_price and price < (mid_line - 3 * std_dev):
                            self.order = self.close()
                            self.entry_price = None
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. 获利：价格回归中轨
                        if price <= mid_line:
                            self.order = self.close()
                            self.entry_price = None
                        
                        # 2. 止损：价格继续上涨，超过3倍标准差
                        elif self.entry_price and price > (mid_line + 3 * std_dev):
                            self.order = self.close()
                            self.entry_price = None
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class BollingerMR_20_2(BollingerMeanReversionAtom):
    """标准布林带回归：20周期，2倍标准差"""
    name = "boll_mr_20_2"
    params = {'period': 20, 'devfactor': 2.0, 'bandwidth_threshold': 0.05, 'risk_pct': 0.1}


class BollingerMR_20_2_5(BollingerMeanReversionAtom):
    """宽幅布林带回归：20周期，2.5倍标准差（更宽带宽）"""
    name = "boll_mr_20_2_5"
    params = {'period': 20, 'devfactor': 2.5, 'bandwidth_threshold': 0.06, 'risk_pct': 0.1}


class BollingerMR_20_1_5(BollingerMeanReversionAtom):
    """窄幅布林带回归：20周期，1.5倍标准差（更窄带宽）"""
    name = "boll_mr_20_1_5"
    params = {'period': 20, 'devfactor': 1.5, 'bandwidth_threshold': 0.04, 'risk_pct': 0.1}


class BollingerMR_30_2(BollingerMeanReversionAtom):
    """平滑布林带回归：30周期，2倍标准差（更平滑）"""
    name = "boll_mr_30_2"
    params = {'period': 30, 'devfactor': 2.0, 'bandwidth_threshold': 0.05, 'risk_pct': 0.1}


class BollingerMR_10_2(BollingerMeanReversionAtom):
    """快速布林带回归：10周期，2倍标准差（更敏感）"""
    name = "boll_mr_10_2"
    params = {'period': 10, 'devfactor': 2.0, 'bandwidth_threshold': 0.05, 'risk_pct': 0.1}


class BollingerMR_Strict(BollingerMeanReversionAtom):
    """严格布林带回归：20周期，2倍标准差，更低带宽阈值"""
    name = "boll_mr_strict"
    params = {'period': 20, 'devfactor': 2.0, 'bandwidth_threshold': 0.03, 'risk_pct': 0.1}
