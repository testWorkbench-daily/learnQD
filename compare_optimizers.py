#!/usr/bin/env python
"""
对比夏普优化器 vs 收益率优化器

用法:
    python compare_optimizers.py --start 20240101 --end 20241231 --timeframe d1
"""

import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description='对比夏普优化器 vs 收益率优化器的结果')
    parser.add_argument('--start', required=True, help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end', required=True, help='结束日期 (YYYYMMDD)')
    parser.add_argument('--timeframe', default='d1', help='时间周期')
    parser.add_argument('--results-dir', default='backtest_results', help='结果目录')
    args = parser.parse_args()

    # 读取两个优化器的结果
    sharpe_file = f"{args.results_dir}/optimized_portfolios_{args.timeframe}_{args.start}_{args.end}.csv"
    return_file = f"{args.results_dir}/max_return_portfolios_{args.timeframe}_{args.start}_{args.end}.csv"

    try:
        sharpe_df = pd.read_csv(sharpe_file)
        return_df = pd.read_csv(return_file)
    except FileNotFoundError as e:
        print(f"错误: 未找到文件 - {e}")
        print("\n请先运行两个优化器:")
        print(f"  python portfolio_optimizer.py --start {args.start} --end {args.end} --timeframe {args.timeframe}")
        print(f"  python max_return_optimizer.py --start {args.start} --end {args.end} --timeframe {args.timeframe}")
        return

    # 对比最佳组合
    print("\n" + "=" * 100)
    print("夏普优化器 vs 收益率优化器 - 最佳组合对比")
    print("=" * 100)

    print("\n【夏普优化器 - 最佳风险调整收益组合】")
    best_sharpe = sharpe_df.iloc[0]
    print(f"  策略组成: {best_sharpe['strategies']}")
    print(f"  权重方法: {best_sharpe['weight_method']}")
    print(f"  权重分配: {best_sharpe['weights']}")
    print(f"  预期夏普: {best_sharpe['expected_sharpe']:.2f}")
    print(f"  预期收益: {best_sharpe['expected_return']*100:.2f}%")
    print(f"  预期回撤: {best_sharpe['expected_max_dd']*100:.2f}%")

    print("\n【收益率优化器 - 最佳绝对收益组合】")
    best_return = return_df.iloc[0]
    print(f"  策略组成: {best_return['strategies']}")
    print(f"  权重方法: {best_return['weight_method']}")
    print(f"  权重分配: {best_return['weights']}")
    print(f"  预期夏普: {best_return['expected_sharpe']:.2f}")
    print(f"  预期收益: {best_return['expected_return']*100:.2f}%")
    print(f"  预期回撤: {best_return['expected_max_dd']*100:.2f}%")

    # 汇总对比
    print("\n" + "=" * 100)
    print("汇总对比")
    print("=" * 100)
    print(f"{'指标':<20} {'夏普优化器':<20} {'收益率优化器':<20} {'差异':<20}")
    print("-" * 100)

    sharpe_val = best_sharpe['expected_sharpe']
    return_sharpe = best_return['expected_sharpe']
    print(f"{'预期夏普比率':<20} {sharpe_val:<20.2f} {return_sharpe:<20.2f} {return_sharpe - sharpe_val:+.2f}")

    sharpe_ret = best_sharpe['expected_return'] * 100
    return_ret = best_return['expected_return'] * 100
    print(f"{'预期收益率%':<20} {sharpe_ret:<20.2f} {return_ret:<20.2f} {return_ret - sharpe_ret:+.2f}")

    sharpe_dd = best_sharpe['expected_max_dd'] * 100
    return_dd = best_return['expected_max_dd'] * 100
    print(f"{'预期最大回撤%':<20} {sharpe_dd:<20.2f} {return_dd:<20.2f} {return_dd - sharpe_dd:+.2f}")

    # 投资建议
    print("\n" + "=" * 100)
    print("投资建议")
    print("=" * 100)

    print("\n【稳健型投资者】")
    print("  推荐: 100% 夏普优化器组合")
    print(f"  理由: 夏普比率 {sharpe_val:.2f}，回撤仅 {sharpe_dd:.2f}%，适合风险厌恶型")

    print("\n【激进型投资者】")
    print("  推荐: 100% 收益率优化器组合")
    print(f"  理由: 收益率 {return_ret:.2f}%，可承受 {return_dd:.2f}% 回撤")

    print("\n【平衡型投资者】")
    avg_sharpe = (sharpe_val + return_sharpe) / 2
    avg_return = (sharpe_ret + return_ret) / 2
    avg_dd = (sharpe_dd + return_dd) / 2
    print("  推荐: 70% 夏普优化 + 30% 收益优化")
    print(f"  预期: 夏普 {avg_sharpe:.2f}, 收益 {avg_return:.2f}%, 回撤 {avg_dd:.2f}%")

    # 前5名对比
    print("\n" + "=" * 100)
    print("前5名组合对比")
    print("=" * 100)

    print("\n夏普优化器 - 前5名:")
    print(f"{'排名':<6} {'预期夏普':<12} {'预期收益%':<12} {'预期回撤%':<12} {'策略数':<8}")
    print("-" * 60)
    for idx, row in sharpe_df.head(5).iterrows():
        print(f"{row['portfolio_id']:<6} {row['expected_sharpe']:<12.2f} "
              f"{row['expected_return']*100:<12.2f} {row['expected_max_dd']*100:<12.2f} {row['num_strategies']:<8}")

    print("\n收益率优化器 - 前5名:")
    print(f"{'排名':<6} {'预期收益%':<12} {'预期夏普':<12} {'预期回撤%':<12} {'策略数':<8}")
    print("-" * 60)
    for idx, row in return_df.head(5).iterrows():
        print(f"{row['portfolio_id']:<6} {row['expected_return']*100:<12.2f} "
              f"{row['expected_sharpe']:<12.2f} {row['expected_max_dd']*100:<12.2f} {row['num_strategies']:<8}")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
