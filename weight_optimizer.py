#!/usr/bin/env python
"""
权重优化器

提供6种权重计算方法:
1. sharpe_weighted - 夏普加权（推荐）
2. risk_parity - 风险平价
3. max_sharpe - 最大夏普比率优化（数学最优解）
4. return_weighted - 收益加权
5. max_return - 最大收益率优化（激进型）
6. equal_weight - 等权重

用法:
    from weight_optimizer import WeightOptimizer
    import pandas as pd

    # returns_df: DataFrame，列为策略名，每行为daily_return
    optimizer = WeightOptimizer(returns_df)
    weights = optimizer.calculate_weights('sharpe_weighted')
"""

import numpy as np
import pandas as pd
from typing import List
from scipy.optimize import minimize


class WeightOptimizer:
    """权重优化器"""

    def __init__(self, returns_df: pd.DataFrame):
        """
        初始化

        Args:
            returns_df: DataFrame，列为策略名，每行为daily_return
        """
        self.returns_df = returns_df
        self.strategies = returns_df.columns.tolist()
        self.n_strategies = len(self.strategies)

    def calculate_weights(self, method: str = 'sharpe_weighted') -> np.ndarray:
        """
        计算权重（统一接口）

        Args:
            method: 权重方法
                - 'sharpe_weighted': 夏普加权（推荐）
                - 'risk_parity': 风险平价
                - 'max_sharpe': 最大夏普比率优化
                - 'return_weighted': 收益加权
                - 'max_return': 最大收益率优化（激进型）
                - 'equal_weight': 等权重

        Returns:
            权重数组，和为1，长度为策略数量
        """
        method = method.lower()

        if method == 'sharpe_weighted':
            return self._sharpe_weighted()
        elif method == 'risk_parity':
            return self._risk_parity()
        elif method == 'max_sharpe':
            return self._max_sharpe_optimization()
        elif method == 'return_weighted':
            return self._return_weighted()
        elif method == 'max_return':
            return self._max_return_optimization()
        elif method == 'equal_weight':
            return self._equal_weight()
        else:
            raise ValueError(f"未知的权重方法: {method}\n"
                           f"可选: sharpe_weighted, risk_parity, max_sharpe, return_weighted, max_return, equal_weight")

    def _sharpe_weighted(self) -> np.ndarray:
        """
        夏普加权（推荐方法）

        步骤:
        1. 计算每个策略的夏普比率
        2. 将负夏普设为0（避免负权重）
        3. 按夏普比率归一化

        公式: w_i = sharpe_i / Σ(sharpe_j)
        """
        sharpes = []
        for col in self.strategies:
            returns = self.returns_df[col]
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0.0
            sharpes.append(max(sharpe, 0))  # 负夏普设为0

        sharpes = np.array(sharpes)

        # 如果所有夏普都为0或负，fallback到等权重
        if sharpes.sum() == 0:
            print("  警告: 所有策略夏普比率≤0，使用等权重")
            return self._equal_weight()

        # 归一化
        weights = sharpes / sharpes.sum()
        return weights

    def _risk_parity(self) -> np.ndarray:
        """
        风险平价

        使每个策略的风险贡献相等

        公式: w_i ∝ 1 / volatility_i
        """
        volatilities = []
        for col in self.strategies:
            vol = self.returns_df[col].std() * np.sqrt(252)
            volatilities.append(max(vol, 0.0001))  # 避免除零

        # 反比例：波动率越高，权重越低
        inv_vols = 1.0 / np.array(volatilities)

        # 归一化
        weights = inv_vols / inv_vols.sum()
        return weights

    def _max_sharpe_optimization(self) -> np.ndarray:
        """
        最大夏普比率优化（数学最优解）

        使用 scipy.optimize 求解:
        max: (μ^T w) / sqrt(w^T Σ w)
        s.t.: Σw_i = 1, w_i ≥ 0

        其中:
        - μ: 各策略平均收益向量
        - Σ: 协方差矩阵
        - w: 权重向量
        """
        mean_returns = self.returns_df.mean().values
        cov_matrix = self.returns_df.cov().values

        def neg_sharpe(weights):
            """负夏普比率（因为scipy最小化）"""
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            if portfolio_vol == 0:
                return 0  # 避免除零

            # 返回负夏普（最小化 = 最大化夏普）
            return -portfolio_return / portfolio_vol

        # 约束条件: 权重和为1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # 边界: 每个权重在 [0, 1]
        bounds = tuple((0, 1) for _ in range(self.n_strategies))

        # 初始猜测：等权重
        init_weights = np.array([1/self.n_strategies] * self.n_strategies)

        # 优化
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            return result.x
        else:
            print(f"  警告: 最大夏普优化失败 ({result.message})，使用夏普加权")
            return self._sharpe_weighted()

    def _return_weighted(self) -> np.ndarray:
        """
        收益加权

        公式: w_i = return_i / Σ(return_j)
        （仅考虑正收益策略）
        """
        total_returns = []
        for col in self.strategies:
            # 计算累积收益率
            total_ret = (1 + self.returns_df[col]).prod() - 1
            total_returns.append(max(total_ret, 0))  # 负收益设为0

        total_returns = np.array(total_returns)

        # 如果所有策略都是负收益，fallback到等权重
        if total_returns.sum() == 0:
            print("  警告: 所有策略收益率≤0，使用等权重")
            return self._equal_weight()

        # 归一化
        weights = total_returns / total_returns.sum()
        return weights

    def _max_return_optimization(self) -> np.ndarray:
        """
        最大收益率优化（激进型，不考虑风险）

        使用 scipy.optimize 求解:
        max: μ^T w
        s.t.: Σw_i = 1, w_i ≥ 0

        其中:
        - μ: 各策略平均收益向量
        - w: 权重向量

        注意：与 max_sharpe 不同，这里不除以波动率，直接最大化收益
        """
        mean_returns = self.returns_df.mean().values

        def neg_return(weights):
            """负收益率（因为scipy最小化）"""
            # 目标：最大化加权平均收益
            # 使用负号因为scipy只能最小化
            return -np.dot(weights, mean_returns)

        # 约束条件: 权重和为1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # 边界: 每个权重在 [0, 1]
        bounds = tuple((0, 1) for _ in range(self.n_strategies))

        # 初始猜测：等权重
        init_weights = np.array([1/self.n_strategies] * self.n_strategies)

        # 优化
        result = minimize(
            neg_return,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if result.success:
            return result.x
        else:
            print(f"  警告: 最大收益优化失败 ({result.message})，使用收益加权")
            return self._return_weighted()

    def _equal_weight(self) -> np.ndarray:
        """等权重"""
        return np.array([1/self.n_strategies] * self.n_strategies)

    def get_portfolio_metrics(self, weights: np.ndarray) -> dict:
        """
        计算组合的预期指标

        Args:
            weights: 权重数组

        Returns:
            {
                'return': 预期收益率,
                'volatility': 预期波动率,
                'sharpe': 预期夏普比率
            }
        """
        # 加权平均收益率
        weighted_returns = (self.returns_df * weights).sum(axis=1)

        # 年化收益率
        mean_return = weighted_returns.mean() * 252

        # 年化波动率
        volatility = weighted_returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe = mean_return / volatility if volatility > 0 else 0.0

        return {
            'return': mean_return,
            'volatility': volatility,
            'sharpe': sharpe
        }

    def compare_methods(self, methods: List[str] = None) -> pd.DataFrame:
        """
        对比不同权重方法的效果

        Args:
            methods: 权重方法列表，默认对比所有6种

        Returns:
            DataFrame，包含各方法的权重和预期指标
        """
        if methods is None:
            methods = ['sharpe_weighted', 'risk_parity', 'max_sharpe', 'return_weighted', 'max_return', 'equal_weight']

        results = []

        for method in methods:
            weights = self.calculate_weights(method)
            metrics = self.get_portfolio_metrics(weights)

            # 构建权重字典
            weight_dict = {f'w_{s}': w for s, w in zip(self.strategies, weights)}

            # 合并
            row = {
                'method': method,
                **metrics,
                **weight_dict
            }
            results.append(row)

        df = pd.DataFrame(results)
        return df


def main():
    """测试用例"""
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')

    # 3个策略的日收益率
    returns_data = {
        'strategy_A': np.random.normal(0.001, 0.02, 252),  # 夏普约0.35
        'strategy_B': np.random.normal(0.0015, 0.015, 252),  # 夏普约0.7
        'strategy_C': np.random.normal(0.0008, 0.025, 252),  # 夏普约0.2
    }

    returns_df = pd.DataFrame(returns_data, index=dates)

    print("=" * 80)
    print("权重优化器测试")
    print("=" * 80)
    print("\n策略日收益率统计:")
    print(returns_df.describe())

    # 创建优化器
    optimizer = WeightOptimizer(returns_df)

    # 对比所有方法
    print("\n\n" + "=" * 80)
    print("权重方法对比")
    print("=" * 80)

    comparison = optimizer.compare_methods()
    print(comparison.to_string(index=False))

    # 单独测试每种方法
    print("\n\n" + "=" * 80)
    print("各方法详细权重")
    print("=" * 80)

    for method in ['sharpe_weighted', 'risk_parity', 'max_sharpe', 'return_weighted', 'max_return', 'equal_weight']:
        print(f"\n{method}:")
        weights = optimizer.calculate_weights(method)
        for s, w in zip(optimizer.strategies, weights):
            print(f"  {s}: {w*100:.2f}%")


if __name__ == '__main__':
    main()
