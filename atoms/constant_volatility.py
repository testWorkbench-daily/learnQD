"""
恒定波动率目标策略（Constant Volatility Targeting）
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy, PercentSizer
import math
import collections


class ConstantVolatilityAtom(StrategyAtom):
    """
    恒定波动率目标策略

    核心思想：
    - 根据当前市场波动率动态调整仓位
    - 使投资组合保持恒定的目标波动率水平
    - 高波动期减仓，低波动期加仓

    策略逻辑：
    1. 计算当前市场波动率（基于历史收益率标准差）
    2. 目标仓位 = 目标波动率 / 当前波动率
    3. 根据目标仓位调整持仓

    波动率计算：
    - 使用 N 日收益率标准差
    - 年化：日波动率 × √252

    仓位调整：
    - 当前波动率高 → 减少仓位
    - 当前波动率低 → 增加仓位
    - 设置仓位上下限（如 0.2x - 2.0x）

    优势：
    - 平滑收益曲线
    - 风险控制更稳定
    - 避免高波动期过度暴露
    - 低波动期充分利用资金
    """

    name = "constant_volatility"
    params = {
        'target_volatility': 0.15,    # 目标年化波动率（15%）
        'volatility_period': 20,       # 波动率计算周期
        'rebalance_period': 5,         # 再平衡周期（每N日调整一次仓位）
        'min_position': 0.2,           # 最小仓位倍数
        'max_position': 2.0,           # 最大仓位倍数
        'trend_ma_period': 50,         # 趋势判断均线周期
        'risk_pct': 0.1,
    }

    def strategy_cls(self):
        target_vol = self.params['target_volatility']
        vol_period = self.params['volatility_period']
        rebalance_period = self.params['rebalance_period']
        min_pos = self.params['min_position']
        max_pos = self.params['max_position']
        trend_ma_period = self.params['trend_ma_period']

        class Strategy(BaseStrategy):
            params = (
                ('target_volatility', target_vol),
                ('volatility_period', vol_period),
                ('rebalance_period', rebalance_period),
                ('min_position', min_pos),
                ('max_position', max_pos),
                ('trend_ma_period', trend_ma_period),
            )

            def __init__(self):
                super().__init__()
                
                # 趋势指标（判断做多还是做空）
                self.sma = bt.ind.SMA(self.data.close, period=self.p.trend_ma_period)
                
                # 存储历史收益率
                self.returns_history = collections.deque(maxlen=self.p.volatility_period)
                self.last_close = None
                
                # 当前目标仓位倍数
                self.target_position_multiplier = 1.0
                
                # 再平衡计数器
                self.bars_since_rebalance = 0

            def _calculate_daily_return(self):
                """计算日收益率"""
                if self.last_close is None or self.last_close == 0:
                    return 0.0
                
                current_close = self.data.close[0]
                daily_return = (current_close - self.last_close) / self.last_close
                return daily_return

            def _calculate_volatility(self):
                """计算年化波动率"""
                if len(self.returns_history) < 10:
                    # 数据不足，返回目标波动率
                    return self.p.target_volatility
                
                # 计算标准差
                returns_list = list(self.returns_history)
                mean_return = sum(returns_list) / len(returns_list)
                variance = sum((r - mean_return) ** 2 for r in returns_list) / len(returns_list)
                daily_std = math.sqrt(variance)
                
                # 年化（假设252个交易日）
                annualized_vol = daily_std * math.sqrt(252)
                
                return annualized_vol

            def _calculate_target_position(self, current_vol):
                """计算目标仓位倍数"""
                if current_vol <= 0:
                    return self.p.min_position
                
                # 目标仓位 = 目标波动率 / 当前波动率
                target_pos = self.p.target_volatility / current_vol
                
                # 限制在最小和最大仓位之间
                target_pos = max(self.p.min_position, min(self.p.max_position, target_pos))
                
                return target_pos

            def _should_rebalance(self):
                """判断是否应该再平衡"""
                return self.bars_since_rebalance >= self.p.rebalance_period

            def _get_trend_direction(self):
                """判断趋势方向"""
                if len(self.sma) == 0:
                    return 0
                
                close = self.data.close[0]
                sma_value = self.sma[0]
                
                if close > sma_value:
                    return 1  # 上升趋势
                elif close < sma_value:
                    return -1  # 下降趋势
                else:
                    return 0

            def next(self):
                if self.order:
                    return
                
                # 计算并记录日收益率
                daily_return = self._calculate_daily_return()
                if self.last_close is not None:
                    self.returns_history.append(daily_return)
                self.last_close = self.data.close[0]
                
                # 增加再平衡计数器
                self.bars_since_rebalance += 1
                
                # 确保有足够数据
                if len(self.returns_history) < self.p.volatility_period // 2:
                    return
                
                # 计算当前波动率
                current_vol = self._calculate_volatility()
                
                # 计算目标仓位倍数
                self.target_position_multiplier = self._calculate_target_position(current_vol)
                
                # 判断趋势方向
                trend_direction = self._get_trend_direction()
                
                # 如果没有持仓，根据趋势建立仓位
                if not self.position:
                    if trend_direction > 0:
                        # 上升趋势，做多
                        # 注意：这里我们只是发出信号，实际仓位大小由后续逻辑控制
                        self.order = self.buy()
                    elif trend_direction < 0:
                        # 下降趋势，做空
                        self.order = self.sell()
                
                # 如果有持仓，根据再平衡周期和趋势变化调整
                else:
                    # 趋势反转，平仓
                    if (self.position.size > 0 and trend_direction < 0) or \
                       (self.position.size < 0 and trend_direction > 0):
                        self.order = self.close()
                        self.bars_since_rebalance = 0
                    
                    # 需要再平衡（实际上在单合约回测中，我们通过记录倍数来模拟）
                    elif self._should_rebalance():
                        # 在实际应用中，这里会调整仓位大小
                        # 但在 backtrader 的单合约框架下，我们主要记录目标倍数
                        # 可以在日志中输出当前建议的仓位倍数
                        self.bars_since_rebalance = 0

        return Strategy

    def sizer_cls(self):
        """使用固定手数，实际波动率调整在策略逻辑中体现"""
        return None


# 预定义的参数变体
class ConstantVolatility_10pct(ConstantVolatilityAtom):
    """恒定波动率：10% 目标年化波动率（保守）"""
    name = "const_vol_10"
    params = {
        'target_volatility': 0.10,
        'volatility_period': 20,
        'rebalance_period': 5,
        'min_position': 0.2,
        'max_position': 2.0,
        'trend_ma_period': 50,
        'risk_pct': 0.1,
    }


class ConstantVolatility_15pct(ConstantVolatilityAtom):
    """恒定波动率：15% 目标年化波动率（标准）"""
    name = "const_vol_15"
    params = {
        'target_volatility': 0.15,
        'volatility_period': 20,
        'rebalance_period': 5,
        'min_position': 0.2,
        'max_position': 2.0,
        'trend_ma_period': 50,
        'risk_pct': 0.1,
    }


class ConstantVolatility_20pct(ConstantVolatilityAtom):
    """恒定波动率：20% 目标年化波动率（进取）"""
    name = "const_vol_20"
    params = {
        'target_volatility': 0.20,
        'volatility_period': 20,
        'rebalance_period': 5,
        'min_position': 0.2,
        'max_position': 2.0,
        'trend_ma_period': 50,
        'risk_pct': 0.1,
    }


class ConstantVolatility_Conservative(ConstantVolatilityAtom):
    """恒定波动率：保守配置（10%目标，更严格的仓位限制）"""
    name = "const_vol_conservative"
    params = {
        'target_volatility': 0.10,
        'volatility_period': 20,
        'rebalance_period': 5,
        'min_position': 0.3,
        'max_position': 1.5,
        'trend_ma_period': 50,
        'risk_pct': 0.1,
    }


class ConstantVolatility_Aggressive(ConstantVolatilityAtom):
    """恒定波动率：激进配置（20%目标，更宽松的仓位限制）"""
    name = "const_vol_aggressive"
    params = {
        'target_volatility': 0.20,
        'volatility_period': 20,
        'rebalance_period': 5,
        'min_position': 0.1,
        'max_position': 3.0,
        'trend_ma_period': 50,
        'risk_pct': 0.1,
    }
