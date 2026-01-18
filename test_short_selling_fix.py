#!/usr/bin/env python
"""测试RSI做空修复效果"""
import datetime
from bt_runner import Runner
from atoms.portfolio_rank3_combo import PortfolioRank3ComboAtom

print("=" * 80)
print("测试RSI做空功能修复")
print("=" * 80)

# 测试2024年全年
runner = Runner(
    data_path='data/nq_m1_forward_adjusted.csv',
    timeframe='d1',
    start_date=datetime.datetime(2024, 1, 1),
    end_date=datetime.datetime(2024, 12, 31),
    cash=100000
)

print("\n运行修复后的策略...")
atom = PortfolioRank3ComboAtom()
result = runner.run(atom, save_trades=True, plot=False)

print("\n" + "=" * 80)
print("关键指标对比")
print("=" * 80)
print(f"\n修复后结果:")
print(f"  收益率: {result['return_pct']:.2f}%")
print(f"  夏普比率: {result['sharpe']:.2f}")
print(f"  最大回撤: {result['max_dd']:.2f}%")
print(f"  卡尔玛比率: {result['calmar']:.2f}")

print("\n提示：运行诊断脚本查看做空详情：")
print("  python diagnose_short_selling.py")
