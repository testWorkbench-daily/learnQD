"""
海龟交易策略 (Turtle Trading Rules) - 仅做多版本

经典趋势跟踪策略，核心规则：
1. 20日唐奇安通道突破入场（仅做多）
2. 基于ATR的仓位管理 (每Unit风险1%)
3. 金字塔加仓 (每0.5N加1 Unit，最多4 Units)
4. 2N ATR止损
5. 10日反向突破出场
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy


class TurtleTradingAtom(StrategyAtom):
    """
    海龟交易策略（仅做多）
    
    入场：收盘价突破20日最高点
    加仓：价格每上涨0.5N，加仓1 Unit（最多4 Units）
    止损：入场价 - 2N，每次加仓后止损跟随上移
    出场：收盘价跌破10日最低点
    """
    
    name = "turtle_trading"
    params = {
        'entry_period': 20,          # 入场突破周期（20日）
        'exit_period': 10,           # 出场突破周期（10日）
        'atr_period': 20,            # ATR计算周期
        'risk_per_unit': 0.01,       # 每个Unit风险（1%）
        'max_units': 4,              # 最大Unit数量
        'pyramid_distance': 0.5,     # 加仓距离（0.5N）
        'stop_loss_atr_multiple': 2, # 止损倍数（2N）
        'point_value': 20,           # 每点价值（NQ=$20/点）
    }
    
    def strategy_cls(self):
        p = self.params
        
        class Strategy(BaseStrategy):
            params = (
                ('entry_period', p['entry_period']),
                ('exit_period', p['exit_period']),
                ('atr_period', p['atr_period']),
                ('risk_per_unit', p['risk_per_unit']),
                ('max_units', p['max_units']),
                ('pyramid_distance', p['pyramid_distance']),
                ('stop_loss_atr_multiple', p['stop_loss_atr_multiple']),
                ('point_value', p['point_value']),
            )
            
            def __init__(self):
                super().__init__()
                
                # === 唐奇安通道 - 入场用 ===
                self.entry_high = bt.indicators.Highest(
                    self.data.high, 
                    period=self.p.entry_period,
                    subplot=False
                )
                self.entry_low = bt.indicators.Lowest(
                    self.data.low, 
                    period=self.p.entry_period,
                    subplot=False
                )
                
                # === 唐奇安通道 - 出场用 ===
                self.exit_high = bt.indicators.Highest(
                    self.data.high, 
                    period=self.p.exit_period,
                    subplot=False
                )
                self.exit_low = bt.indicators.Lowest(
                    self.data.low, 
                    period=self.p.exit_period,
                    subplot=False
                )
                
                # === ATR指标 (N值) ===
                self.atr = bt.indicators.ATR(
                    self.data, 
                    period=self.p.atr_period
                )
                
                # === 状态变量 ===
                self.units_held = 0            # 当前持有Unit数量
                self.last_add_price = None     # 上次加仓价格
                self.stop_loss = None          # 当前止损价位
                self.entry_price = None        # 初始入场价格
                
                # === 交易统计 ===
                self.turtle_trade_count = 0    # 交易次数
                self.turtle_win_count = 0      # 盈利交易数
            
            def calculate_unit_size(self):
                """
                计算单位头寸大小
                
                公式: Unit = (账户资金 × 风险百分比) / (N × 每点价值)
                """
                N = self.atr[0]
                if N <= 0:
                    return 0
                
                account_value = self.broker.getvalue()
                point_value = self.p.point_value
                
                # 计算Unit大小
                unit_size = (account_value * self.p.risk_per_unit) / (N * point_value)
                unit_size = int(unit_size)
                
                # 最小1手
                return max(1, unit_size)
            
            def _reset_state(self):
                """重置持仓状态"""
                self.units_held = 0
                self.last_add_price = None
                self.stop_loss = None
                self.entry_price = None
            
            def next(self):
                # 如果有未完成订单，等待
                if self.order:
                    return
                
                # 获取当前数据
                current_close = self.data.close[0]
                current_low = self.data.low[0]
                N = self.atr[0]
                
                # ATR未就绪，跳过
                if N <= 0:
                    return
                
                # 计算Unit大小
                unit_size = self.calculate_unit_size()
                if unit_size <= 0:
                    return
                
                # ========================================
                # 情况1: 无持仓 - 寻找做多入场信号
                # ========================================
                if not self.position:
                    
                    # === 做多入场信号 ===
                    # 条件: 收盘价突破20日最高点
                    if current_close > self.entry_high[-1]:
                        self.order = self.buy(size=unit_size)
                        
                        # 更新状态
                        self.units_held = 1
                        self.entry_price = current_close
                        self.last_add_price = current_close
                        self.stop_loss = current_close - self.p.stop_loss_atr_multiple * N
                        
                        self.log(f'【做多入场】价格={current_close:.2f}, '
                                f'数量={unit_size}手, '
                                f'止损={self.stop_loss:.2f}, N={N:.2f}')
                
                # ========================================
                # 情况2: 有多头持仓 - 管理持仓
                # ========================================
                elif self.position.size > 0:
                    
                    # 防御性检查：如果状态变量未初始化，跳过本次
                    if self.stop_loss is None or self.last_add_price is None:
                        return
                    
                    # 【优先级1】止损检查
                    if current_low <= self.stop_loss:
                        self.order = self.close()
                        self.log(f'【多头止损】价格={current_close:.2f}, '
                                f'止损位={self.stop_loss:.2f}, '
                                f'Units={self.units_held}')
                        self._reset_state()
                        return
                    
                    # 【优先级2】出场信号检查
                    # 条件: 收盘价跌破10日最低点
                    if current_close < self.exit_low[-1]:
                        self.order = self.close()
                        self.log(f'【多头出场】价格={current_close:.2f}, '
                                f'10日低点={self.exit_low[-1]:.2f}, '
                                f'Units={self.units_held}')
                        self._reset_state()
                        return
                    
                    # 【优先级3】金字塔加仓检查
                    if (self.units_held < self.p.max_units and 
                        current_close >= self.last_add_price + self.p.pyramid_distance * N):
                        
                        # 执行加仓
                        self.order = self.buy(size=unit_size)
                        self.units_held += 1
                        self.last_add_price = current_close
                        
                        # 止损上移
                        self.stop_loss = current_close - self.p.stop_loss_atr_multiple * N
                        
                        self.log(f'【多头加仓】Unit {self.units_held}/4, '
                                f'价格={current_close:.2f}, '
                                f'新止损={self.stop_loss:.2f}')
            
            def stop(self):
                """回测结束时输出统计"""
                final_value = self.broker.getvalue()
                self.log(f'【回测完成】最终资金: {final_value:.2f}')
        
        return Strategy


# =============================================================================
# 参数变体 - 系统1（20日突破，标准参数）
# =============================================================================
class Turtle_System1_Standard(TurtleTradingAtom):
    """系统1标准参数：20日入场，10日出场，1%风险"""
    name = "turtle_sys1_std"
    params = {
        'entry_period': 20,
        'exit_period': 10,
        'atr_period': 20,
        'risk_per_unit': 0.01,
        'max_units': 4,
        'pyramid_distance': 0.5,
        'stop_loss_atr_multiple': 2,
        'point_value': 20,
    }


class Turtle_System1_Conservative(TurtleTradingAtom):
    """系统1保守参数：降低风险至0.5%"""
    name = "turtle_sys1_conservative"
    params = {
        'entry_period': 20,
        'exit_period': 10,
        'atr_period': 20,
        'risk_per_unit': 0.005,        # 降低风险
        'max_units': 3,                 # 减少最大单元数
        'pyramid_distance': 0.75,       # 增加加仓间距
        'stop_loss_atr_multiple': 2.5,  # 稍宽止损
        'point_value': 20,
    }


class Turtle_System1_Aggressive(TurtleTradingAtom):
    """系统1激进参数：提高风险至2%"""
    name = "turtle_sys1_aggressive"
    params = {
        'entry_period': 20,
        'exit_period': 10,
        'atr_period': 20,
        'risk_per_unit': 0.02,         # 提高风险
        'max_units': 4,
        'pyramid_distance': 0.5,
        'stop_loss_atr_multiple': 1.5, # 更紧止损
        'point_value': 20,
    }


# =============================================================================
# 参数变体 - 系统2（55日突破，长期趋势）
# =============================================================================
class Turtle_System2_Standard(TurtleTradingAtom):
    """系统2标准参数：55日入场，20日出场（长期趋势）"""
    name = "turtle_sys2_std"
    params = {
        'entry_period': 55,            # 长期突破
        'exit_period': 20,             # 较宽出场
        'atr_period': 20,
        'risk_per_unit': 0.01,
        'max_units': 4,
        'pyramid_distance': 0.5,
        'stop_loss_atr_multiple': 2,
        'point_value': 20,
    }


# =============================================================================
# 参数变体 - 不同每点价值（适配其他品种）
# =============================================================================
class Turtle_ES_Futures(TurtleTradingAtom):
    """ES期货（标普500期货）：每点$50"""
    name = "turtle_es"
    params = {
        'entry_period': 20,
        'exit_period': 10,
        'atr_period': 20,
        'risk_per_unit': 0.01,
        'max_units': 4,
        'pyramid_distance': 0.5,
        'stop_loss_atr_multiple': 2,
        'point_value': 50,             # ES每点$50
    }


class Turtle_MNQ_Micro(TurtleTradingAtom):
    """微型NQ期货（MNQ）：每点$2"""
    name = "turtle_mnq"
    params = {
        'entry_period': 20,
        'exit_period': 10,
        'atr_period': 20,
        'risk_per_unit': 0.01,
        'max_units': 4,
        'pyramid_distance': 0.5,
        'stop_loss_atr_multiple': 2,
        'point_value': 2,              # MNQ每点$2
    }

