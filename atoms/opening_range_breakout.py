"""
开盘区间突破策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer
import datetime


class OpeningRangeBreakoutAtom(StrategyAtom):
    """
    开盘区间突破策略（ORB - Opening Range Breakout）
    
    开盘后一段时间确定当日波动区间，突破该区间顺势交易。
    假设开盘阶段包含重要信息，反映市场当日情绪。
    
    核心原理：
    - 开盘区间 = 开盘后N个bar的最高价和最低价
    - 区间代表当日初期波动范围
    - 突破区间表示趋势启动
    
    信号规则：
    - 记录开盘后N个bar形成的区间
    - 做多：价格向上突破区间上沿
    - 做空：价格向下跌破区间下沿
    - 当日收盘前平仓（可选）
    
    止损设置：
    - 止损位：区间另一侧或区间中点
    - 盈亏比目标：至少1:1.5
    
    特点：
    - 日内交易策略
    - 需要分钟级数据
    - 适合股指期货、商品期货
    - 简单有效，易于执行
    """
    
    name = "opening_range_breakout"
    params = {'range_bars': 30, 'close_eod': True, 'risk_pct': 0.1}
    
    def strategy_cls(self):
        range_bars = self.params['range_bars']
        close_eod = self.params['close_eod']
        
        class Strategy(BaseStrategy):
            params = (
                ('range_bars', range_bars),      # 开盘区间bar数（如30分钟用30个1分钟bar）
                ('close_eod', close_eod),        # 是否收盘前平仓
            )
            
            def __init__(self):
                super().__init__()
                # 记录当前日期
                self.current_date = None
                # 开盘区间
                self.range_high = None
                self.range_low = None
                self.range_formed = False
                # 当日bar计数
                self.bars_today = 0
                # 入场价格和止损
                self.entry_price = None
                self.stop_loss = None
            
            def next(self):
                if self.order:
                    return
                
                # 获取当前日期
                current_dt = self.data.datetime.datetime(0)
                today = current_dt.date()
                
                # 检查是否是新的一天
                if self.current_date != today:
                    # 新的一天，重置所有变量
                    self.current_date = today
                    self.range_high = None
                    self.range_low = None
                    self.range_formed = False
                    self.bars_today = 0
                
                # 增加当日bar计数
                self.bars_today += 1
                
                # 形成开盘区间
                if not self.range_formed:
                    if self.bars_today <= self.p.range_bars:
                        # 在区间形成期间，记录最高和最低价
                        if self.range_high is None:
                            self.range_high = self.data.high[0]
                            self.range_low = self.data.low[0]
                        else:
                            self.range_high = max(self.range_high, self.data.high[0])
                            self.range_low = min(self.range_low, self.data.low[0])
                        
                        # 区间形成完毕
                        if self.bars_today == self.p.range_bars:
                            self.range_formed = True
                    
                    return  # 区间未形成，不交易
                
                # 区间已形成，开始交易
                current_price = self.data.close[0]
                current_high = self.data.high[0]
                current_low = self.data.low[0]
                
                # 计算区间中点
                range_mid = (self.range_high + self.range_low) / 2.0
                range_size = self.range_high - self.range_low
                
                if not self.position:
                    # 做多信号：价格突破区间上沿
                    if current_high > self.range_high:
                        self.order = self.buy()
                        self.entry_price = current_price
                        # 止损设在区间中点
                        self.stop_loss = range_mid
                    
                    # 做空信号：价格跌破区间下沿
                    elif current_low < self.range_low:
                        self.order = self.sell()
                        self.entry_price = current_price
                        # 止损设在区间中点
                        self.stop_loss = range_mid
                
                else:
                    # 收盘前平仓（如果启用）
                    if self.p.close_eod:
                        # 简化判断：检查是否接近收盘（可根据实际情况调整）
                        # 这里假设14:45之后平仓（针对中国市场）
                        if current_dt.hour >= 14 and current_dt.minute >= 45:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                            return
                    
                    # 平多仓条件
                    if self.position.size > 0:
                        # 1. 止损：价格跌破止损线
                        if current_price <= self.stop_loss:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                        
                        # 2. 获利：达到盈亏比1:1.5
                        elif self.entry_price and current_price >= (self.entry_price + range_size * 1.5):
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                        
                        # 3. 反向突破：跌破区间下沿
                        elif current_low < self.range_low:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                    
                    # 平空仓条件
                    elif self.position.size < 0:
                        # 1. 止损：价格突破止损线
                        if current_price >= self.stop_loss:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                        
                        # 2. 获利：达到盈亏比1:1.5
                        elif self.entry_price and current_price <= (self.entry_price - range_size * 1.5):
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
                        
                        # 3. 反向突破：突破区间上沿
                        elif current_high > self.range_high:
                            self.order = self.close()
                            self.entry_price = None
                            self.stop_loss = None
        
        return Strategy
    
    def sizer_cls(self):
        """使用固定1手"""
        return None  # 使用Runner默认的FixedSize


# 预定义的参数变体
class ORB_15min(OpeningRangeBreakoutAtom):
    """15分钟开盘区间突破（适合1分钟数据）"""
    name = "orb_15min"
    params = {'range_bars': 15, 'close_eod': True, 'risk_pct': 0.1}


class ORB_30min(OpeningRangeBreakoutAtom):
    """30分钟开盘区间突破（适合1分钟数据）"""
    name = "orb_30min"
    params = {'range_bars': 30, 'close_eod': True, 'risk_pct': 0.1}


class ORB_60min(OpeningRangeBreakoutAtom):
    """60分钟开盘区间突破（适合1分钟数据）"""
    name = "orb_60min"
    params = {'range_bars': 60, 'close_eod': True, 'risk_pct': 0.1}


class ORB_30min_NoClose(OpeningRangeBreakoutAtom):
    """30分钟开盘区间突破，不强制收盘平仓"""
    name = "orb_30min_no_close"
    params = {'range_bars': 30, 'close_eod': False, 'risk_pct': 0.1}


class ORB_45min(OpeningRangeBreakoutAtom):
    """45分钟开盘区间突破"""
    name = "orb_45min"
    params = {'range_bars': 45, 'close_eod': True, 'risk_pct': 0.1}


class ORB_Aggressive(OpeningRangeBreakoutAtom):
    """激进ORB：15分钟区间"""
    name = "orb_aggressive"
    params = {'range_bars': 15, 'close_eod': True, 'risk_pct': 0.1}


class ORB_Conservative(OpeningRangeBreakoutAtom):
    """保守ORB：60分钟区间"""
    name = "orb_conservative"
    params = {'range_bars': 60, 'close_eod': True, 'risk_pct': 0.1}
