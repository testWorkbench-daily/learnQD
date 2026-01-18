#!/usr/bin/env python
"""测试基准对比功能"""
import datetime
from bt_runner import Runner
from atoms.portfolio_rank3_combo import PortfolioRank3ComboAtom

# 创建runner（测试2024年全年）
runner = Runner(
    data_path='data/nq_m1_forward_adjusted.csv',
    timeframe='d1',
    start_date=datetime.datetime(2024, 1, 1),
    end_date=datetime.datetime(2024, 12, 31),
    cash=100000
)

# 运行策略
print("开始测试基准对比功能...")
atom = PortfolioRank3ComboAtom()
result = runner.run(atom, save_trades=False, plot=False)

print("\n✅ 测试完成!")
print(f"\n关键指标:")
print(f"  策略收益: {result['return_pct']:.2f}%")
print(f"  年化收益: {result['annualized_return']:.2f}%")
print(f"  年化波动率: {result['volatility']:.2f}%")
print(f"  夏普比率: {result['sharpe']:.2f}")
print(f"  卡尔玛比率: {result['calmar']:.2f}")
