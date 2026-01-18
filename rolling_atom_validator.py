#!/usr/bin/env python3
"""
组合Atom滚动窗口验证器 - 实际执行 vs 理论测试对比

功能：
- 对组合atom在多个时间窗口运行实际回测
- 对同一组合计算理论测试结果（多账户加权）
- 对比4个核心指标：收益率、夏普比率、交易频率、最大回撤
- 生成详细的对比分析报告

用法示例：
    # 验证单个atom
    python rolling_atom_validator.py --atom portfolio_rank3_combo --timeframe d1

    # 自定义时间窗口
    python rolling_atom_validator.py --atom portfolio_rank3_combo --window-months 12 --step-months 6

作者：Claude Code
日期：2026-01-17
"""

import argparse
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp

import pandas as pd
import numpy as np

# 导入依赖模块
from bt_main import ATOMS
from bt_runner import Runner
from portfolio_backtest import backtest_portfolio_from_daily_values

warnings.filterwarnings('ignore')


# ========== Atom配置注册表 ==========
# 手动维护各组合atom的子策略和权重信息
ATOM_CONFIGS = {
    'portfolio_rank3_combo': {
        'strategies': ['vol_breakout_aggressive', 'vol_regime_long', 'triple_ma', 'rsi_reversal'],
        'weights': [0.0843, 0.2390, 0.3366, 0.3401],
        'description': '稳健排名 #3: 波动率突破+波动率择时+三重均线+RSI反转',
    },
    # 可以在这里添加更多组合atom的配置
    # 'portfolio_rank1_combo': {...},
}


class RollingAtomValidator:
    """组合atom滚动窗口验证器 - 实际执行 vs 理论测试"""

    def __init__(
        self,
        atom_name: str,
        timeframe: str = 'd1',
        window_months: int = 12,
        step_months: int = 12,
        data_path: str = './data/nq_m1_forward_adjusted.csv',
        results_dir: str = 'backtest_results',
        output_dir: str = 'backtest_results/atom_validation',
        n_workers: int = 1,
    ):
        """
        初始化验证器

        Args:
            atom_name: atom注册名（如"portfolio_rank3_combo"）
            timeframe: 时间周期（d1/h1/m15等）
            window_months: 窗口长度（月）
            step_months: 滑动步长（月）
            data_path: 数据文件路径
            results_dir: 回测结果目录
            output_dir: 输出目录
            n_workers: 并行进程数
        """
        self.atom_name = atom_name
        self.timeframe = timeframe
        self.window_months = window_months
        self.step_months = step_months
        self.data_path = data_path
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.n_workers = n_workers

        # 验证atom是否注册
        if atom_name not in ATOMS:
            raise ValueError(f"Atom '{atom_name}' 未在bt_main.ATOMS中注册")

        # 验证atom配置是否存在
        if atom_name not in ATOM_CONFIGS:
            raise ValueError(
                f"Atom '{atom_name}' 未在ATOM_CONFIGS中配置\n"
                f"请在rolling_atom_validator.py中添加该atom的配置信息（strategies和weights）"
            )

        # 获取atom配置
        self.atom_config = ATOM_CONFIGS[atom_name]
        self.strategies = self.atom_config['strategies']
        self.weights = self.atom_config['weights']

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 检测数据范围
        self.data_start, self.data_end = self._detect_data_range()

        print(f"初始化验证器:")
        print(f"  Atom: {atom_name}")
        print(f"  描述: {self.atom_config.get('description', 'N/A')}")
        print(f"  子策略: {', '.join(self.strategies)}")
        print(f"  权重: {[f'{w:.2%}' for w in self.weights]}")
        print(f"  数据范围: {self.data_start[:4]}-{self.data_start[4:6]} ~ {self.data_end[:4]}-{self.data_end[4:6]}")
        print("")

    def _detect_data_range(self) -> Tuple[str, str]:
        """检测可用数据的时间范围"""
        import glob

        # 检测单策略daily_values文件的时间范围
        pattern = f"{self.results_dir}/daily_values_*_{self.timeframe}_*.csv"
        files = glob.glob(pattern)

        if not files:
            print(f"警告: 未找到任何 daily_values 文件，使用默认范围")
            return '20200101', '20241231'

        # 从文件名中提取日期
        dates = []
        for file in files:
            parts = Path(file).stem.split('_')
            if len(parts) >= 5:
                try:
                    start_date = parts[-2]
                    end_date = parts[-1]
                    dates.append((start_date, end_date))
                except:
                    continue

        if not dates:
            return '20200101', '20241231'

        all_starts = [d[0] for d in dates]
        all_ends = [d[1] for d in dates]

        return min(all_starts), max(all_ends)

    def generate_windows(self) -> List[Dict]:
        """生成所有滚动窗口（与rolling_portfolio_validator相同逻辑）"""
        windows = []
        window_id = 1

        start = datetime.strptime(self.data_start, '%Y%m%d')
        end = datetime.strptime(self.data_end, '%Y%m%d')

        current_start = start
        while True:
            window_end = current_start + timedelta(days=30 * self.window_months)

            if window_end > end:
                break

            windows.append({
                'window_id': window_id,
                'start_date': current_start.strftime('%Y%m%d'),
                'end_date': window_end.strftime('%Y%m%d'),
                'start_dt': current_start,
                'end_dt': window_end
            })

            window_id += 1
            current_start = current_start + timedelta(days=30 * self.step_months)

        return windows

    def run_atom_backtest(self, window_start: str, window_end: str) -> Dict:
        """
        在指定窗口运行atom实际回测

        Args:
            window_start: 开始日期 (YYYYMMDD)
            window_end: 结束日期 (YYYYMMDD)

        Returns:
            包含回测结果的字典
        """
        atom_start = time.time()

        # 转换日期格式
        start_dt = datetime.strptime(window_start, '%Y%m%d')
        end_dt = datetime.strptime(window_end, '%Y%m%d')

        # 创建Runner
        runner = Runner(
            data_path=self.data_path,
            timeframe=self.timeframe,
            start_date=start_dt,
            end_date=end_dt
        )

        # 获取atom实例
        atom_cls = ATOMS[self.atom_name]
        atom = atom_cls()

        # 运行回测（不保存交易记录，不绘图）
        result = runner.run(atom, save_trades=True, plot=False)

        atom_time = time.time() - atom_start
        print(f'      (实际回测耗时: {atom_time:.1f}秒)')

        # 提取关键指标
        return {
            'return_pct': result['return_pct'],
            'sharpe': result['sharpe'],
            'max_dd': result['max_dd'],
            'total_trades': result['total_trades'],
            'win_rate': result.get('win_rate', 0.0),
            'final_value': result['final_value'],
        }

    def calculate_theory_result(self, window_start: str, window_end: str) -> Dict:
        """
        计算理论测试结果（多账户加权）

        Args:
            window_start: 开始日期 (YYYYMMDD)
            window_end: 结束日期 (YYYYMMDD)

        Returns:
            包含理论测试结果的字典
        """
        theory_start = time.time()

        result = backtest_portfolio_from_daily_values(
            strategies=self.strategies,
            weights=self.weights,
            start_date=window_start,
            end_date=window_end,
            timeframe=self.timeframe,
            results_dir=self.results_dir
        )

        theory_time = time.time() - theory_start
        print(f'      (理论计算耗时: {theory_time:.2f}秒)')

        return {
            'return_pct': result['return_pct'],
            'sharpe': result['sharpe'],
            'max_dd': result['max_dd'],
            'total_trades': result['total_trades'],
            'final_value': result['final_value'],
        }

    def compare_metrics(self, actual: Dict, theory: Dict, window_id: int, window_start: str, window_end: str) -> Dict:
        """
        对比实际vs理论的4个核心指标

        Args:
            actual: 实际回测结果
            theory: 理论测试结果
            window_id: 窗口ID
            window_start: 窗口开始日期
            window_end: 窗口结束日期

        Returns:
            对比结果字典
        """
        # 计算绝对差异
        return_diff = actual['return_pct'] - theory['return_pct']
        sharpe_diff = actual['sharpe'] - theory['sharpe']
        trades_diff = actual['total_trades'] - theory['total_trades']
        max_dd_diff = actual['max_dd'] - theory['max_dd']

        # 计算相对差异（百分比）
        def safe_pct_diff(actual_val, theory_val):
            if abs(theory_val) < 0.01:  # 避免除以接近0的数
                return 0.0
            return (actual_val - theory_val) / abs(theory_val) * 100

        return_diff_pct = safe_pct_diff(actual['return_pct'], theory['return_pct'])
        sharpe_diff_pct = safe_pct_diff(actual['sharpe'], theory['sharpe'])
        trades_diff_pct = safe_pct_diff(actual['total_trades'], theory['total_trades'])
        max_dd_diff_pct = safe_pct_diff(actual['max_dd'], theory['max_dd'])

        return {
            'window_id': window_id,
            'window_start': window_start,
            'window_end': window_end,
            # 实际执行结果
            'actual_return': actual['return_pct'],
            'actual_sharpe': actual['sharpe'],
            'actual_trades': actual['total_trades'],
            'actual_max_dd': actual['max_dd'],
            'actual_win_rate': actual.get('win_rate', 0.0),
            # 理论测试结果
            'theory_return': theory['return_pct'],
            'theory_sharpe': theory['sharpe'],
            'theory_trades': theory['total_trades'],
            'theory_max_dd': theory['max_dd'],
            # 绝对差异
            'return_diff': return_diff,
            'sharpe_diff': sharpe_diff,
            'trades_diff': trades_diff,
            'max_dd_diff': max_dd_diff,
            # 相对差异（百分比）
            'return_diff_pct': return_diff_pct,
            'sharpe_diff_pct': sharpe_diff_pct,
            'trades_diff_pct': trades_diff_pct,
            'max_dd_diff_pct': max_dd_diff_pct,
        }

    def run_validation(self, use_parallel: bool = True) -> List[Dict]:
        """
        运行完整的滚动窗口验证

        Args:
            use_parallel: 是否使用并行处理

        Returns:
            所有窗口的对比结果列表
        """
        print("=" * 80)
        print(f"组合Atom滚动窗口验证: {self.atom_name} ({self.timeframe})")
        print("=" * 80)
        print(f"配置:")
        print(f"  窗口长度: {self.window_months}个月")
        print(f"  滑动步长: {self.step_months}个月")
        print("=" * 80)
        print("")

        # 生成窗口
        print("生成时间窗口...")
        windows = self.generate_windows()
        print(f"  共生成 {len(windows)} 个滚动窗口")
        print("")

        if len(windows) == 0:
            print("错误: 无法生成窗口")
            return []

        start_time = time.time()

        if use_parallel and self.n_workers > 1:
            all_results = self._run_parallel(windows)
        else:
            all_results = self._run_serial(windows)

        elapsed = time.time() - start_time

        print("")
        print(f"验证完成！总耗时: {elapsed:.0f}秒")
        print("")

        return all_results

    def _run_serial(self, windows: List[Dict]) -> List[Dict]:
        """串行执行验证"""
        all_results = []

        for i, window in enumerate(windows):
            window_start_time = time.time()
            window_id = window['window_id']
            start_date = window['start_date']
            end_date = window['end_date']

            print(f"窗口 {i+1}/{len(windows)}: {start_date[:4]}-{start_date[4:6]} ~ {end_date[:4]}-{end_date[4:6]}")

            try:
                # 1. 运行实际回测
                print(f"  [1/2] 运行实际回测...")
                actual_result = self.run_atom_backtest(start_date, end_date)

                # 2. 计算理论结果
                print(f"  [2/2] 计算理论测试...")
                theory_result = self.calculate_theory_result(start_date, end_date)

                # 3. 对比指标
                comparison = self.compare_metrics(actual_result, theory_result, window_id, start_date, end_date)
                all_results.append(comparison)

                # 4. 打印简要结果
                window_time = time.time() - window_start_time
                print(f"  实际: 收益{actual_result['return_pct']:.2f}%, 夏普{actual_result['sharpe']:.2f}, 交易{actual_result['total_trades']}次")
                print(f"  理论: 收益{theory_result['return_pct']:.2f}%, 夏普{theory_result['sharpe']:.2f}, 交易{theory_result['total_trades']}次")
                print(f"  差异: 收益{comparison['return_diff']:.2f}%, 夏普{comparison['sharpe_diff']:.2f}, 交易{comparison['trades_diff']}次")
                print(f"  ⏱️  窗口总耗时: {window_time:.1f}秒")
                print("")

            except Exception as e:
                print(f"  错误: {str(e)}")
                import traceback
                traceback.print_exc()
                print("")
                continue

        return all_results

    def _run_parallel(self, windows: List[Dict]) -> List[Dict]:
        """并行执行验证"""
        print(f"启动并行执行（{self.n_workers}核）...")
        print("")

        args_list = [(self, window, i+1, len(windows)) for i, window in enumerate(windows)]

        with mp.Pool(processes=self.n_workers) as pool:
            all_results = pool.starmap(_run_window_worker, args_list)

        # 过滤掉None结果（失败的窗口）
        all_results = [r for r in all_results if r is not None]

        return all_results

    def generate_report(self, results: List[Dict]):
        """
        生成详细的对比报告

        Args:
            results: 所有窗口的对比结果
        """
        if len(results) == 0:
            print("无有效结果，跳过报告生成")
            return

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # ========== 1. 保存详细CSV ==========
        csv_path = f"{self.output_dir}/atom_vs_theory_comparison_{self.atom_name}_{self.timeframe}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"详细对比已保存: {csv_path}")

        # ========== 2. 计算汇总统计 ==========
        avg_return_diff = df['return_diff'].mean()
        avg_sharpe_diff = df['sharpe_diff'].mean()
        avg_trades_diff = df['trades_diff'].mean()
        avg_max_dd_diff = df['max_dd_diff'].mean()

        avg_return_diff_pct = df['return_diff_pct'].mean()
        avg_sharpe_diff_pct = df['sharpe_diff_pct'].mean()
        avg_trades_diff_pct = df['trades_diff_pct'].mean()

        # ========== 3. 打印终端报告 ==========
        print("")
        print("=" * 80)
        print(f"组合Atom验证报告: {self.atom_name} ({self.timeframe})")
        print("=" * 80)
        print("")

        # 逐窗口详细对比
        for _, row in df.iterrows():
            print(f"【窗口{row['window_id']}】{row['window_start'][:4]}-{row['window_start'][4:6]} ~ {row['window_end'][:4]}-{row['window_end'][4:6]}")
            print(f"  实际执行: 收益率={row['actual_return']:.2f}%, 夏普={row['actual_sharpe']:.2f}, 交易次数={int(row['actual_trades'])}, 最大回撤={row['actual_max_dd']:.2f}%")
            print(f"  理论测试: 收益率={row['theory_return']:.2f}%, 夏普={row['theory_sharpe']:.2f}, 交易次数={int(row['theory_trades'])}, 最大回撤={row['theory_max_dd']:.2f}%")
            print(f"  差异分析:")

            # 收益率差异
            return_symbol = "✓" if row['return_diff'] >= 0 else "✗"
            print(f"    {return_symbol} 收益率: {row['return_diff']:+.2f}% ({row['return_diff_pct']:+.1f}%)  {'实际更优' if row['return_diff'] > 0 else '理论更优'}")

            # 夏普差异
            sharpe_symbol = "✓" if row['sharpe_diff'] >= 0 else "✗"
            print(f"    {sharpe_symbol} 夏普比率: {row['sharpe_diff']:+.2f} ({row['sharpe_diff_pct']:+.1f}%)  {'实际更优' if row['sharpe_diff'] > 0 else '理论更优'}")

            # 交易频率差异
            trades_symbol = "✓" if row['trades_diff'] < 0 else "✗"  # 交易次数少是好事
            print(f"    {trades_symbol} 交易频率: {row['trades_diff']:+.0f}次 ({row['trades_diff_pct']:+.1f}%)  {'实际更少（成本更低）' if row['trades_diff'] < 0 else '实际更多'}")

            # 回撤差异
            dd_symbol = "✓" if row['max_dd_diff'] > 0 else "✗"  # 回撤小（接近0）是好事
            print(f"    {dd_symbol} 最大回撤: {row['max_dd_diff']:+.2f}% ({row['max_dd_diff_pct']:+.1f}%)  {'实际更好（风险更低）' if row['max_dd_diff'] > 0 else '理论更好'}")
            print("")

        # ========== 4. 汇总统计 ==========
        print("=" * 80)
        print(f"【汇总统计】跨{len(results)}个窗口")
        print("=" * 80)
        print("平均差异:")
        print(f"  - 收益率差异: {avg_return_diff:+.2f}% (实际比理论{'高' if avg_return_diff > 0 else '低'}{abs(avg_return_diff):.2f}%)")
        print(f"  - 夏普比率差异: {avg_sharpe_diff:+.2f} (实际比理论{'高' if avg_sharpe_diff > 0 else '低'}{abs(avg_sharpe_diff):.2f})")
        print(f"  - 交易频率差异: {avg_trades_diff:+.0f}次/窗口 (实际{'减少' if avg_trades_diff < 0 else '增加'}{abs(avg_trades_diff_pct):.1f}%交易)")
        print(f"  - 回撤差异: {avg_max_dd_diff:+.2f}% (实际回撤{'更小' if avg_max_dd_diff > 0 else '更大'})")
        print("")

        # ========== 5. 关键洞察 ==========
        print("关键洞察:")

        # 交易频率洞察
        if avg_trades_diff < 0:
            print(f"  ✓ 单账户执行减少了{abs(avg_trades_diff_pct):.0f}%的交易次数（降低成本）")
        else:
            print(f"  ⚠ 单账户执行增加了{avg_trades_diff_pct:.0f}%的交易次数")

        # 收益率洞察
        if abs(avg_return_diff) <= 2.0:
            print(f"  ✓ 收益率差异在可接受范围内（平均{avg_return_diff:+.2f}%）")
        elif avg_return_diff > 0:
            print(f"  ⭐ 实际执行收益率显著更高（平均+{avg_return_diff:.2f}%）")
        else:
            print(f"  ⚠ 实际执行收益率略有下降（平均{avg_return_diff:.2f}%）")

        # 风险洞察
        if avg_max_dd_diff > 0:
            print(f"  ✓ 风险特征略有改善（回撤平均减少{abs(avg_max_dd_diff):.2f}%）")
        else:
            print(f"  ⚠ 风险特征略有恶化（回撤平均增加{abs(avg_max_dd_diff):.2f}%）")

        # 夏普洞察
        if abs(avg_sharpe_diff) <= 0.1:
            print(f"  ✓ 夏普比率基本一致（差异{avg_sharpe_diff:+.2f}）")
        elif avg_sharpe_diff > 0:
            print(f"  ⭐ 夏普比率显著提升（平均+{avg_sharpe_diff:.2f}）")
        else:
            print(f"  ⚠ 夏普比率小幅下降（平均{avg_sharpe_diff:.2f}），主要由于{'交易频率降低' if avg_trades_diff < 0 else '其他因素'}")

        print("")

        # ========== 6. 建议 ==========
        print("建议:")
        if avg_trades_diff < -20 and abs(avg_return_diff) <= 3.0:
            print("  ✓ 该组合适合单账户实际执行")
            print("  ✓ 交易成本节省可能抵消部分收益下降")
        elif avg_return_diff > 2.0:
            print("  ⭐ 该组合在单账户执行下表现更优，强烈推荐")
        elif avg_return_diff < -5.0:
            print("  ⚠ 实际执行与理论测试差异较大，建议进一步分析")
        else:
            print("  ✓ 实际执行与理论测试基本一致")

        print("  - 建议在实盘前进一步测试手续费影响")
        print("  - 可以通过调整仓位阈值优化交易频率")
        print("")

        # ========== 7. 生成汇总文本报告（可选）==========
        summary_path = f"{self.output_dir}/atom_vs_theory_summary_{self.atom_name}_{self.timeframe}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"组合Atom验证报告: {self.atom_name} ({self.timeframe})\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"子策略: {', '.join(self.strategies)}\n")
            f.write(f"权重: {self.weights}\n\n")
            f.write(f"验证窗口数: {len(results)}\n\n")
            f.write("平均差异:\n")
            f.write(f"  收益率: {avg_return_diff:+.2f}%\n")
            f.write(f"  夏普比率: {avg_sharpe_diff:+.2f}\n")
            f.write(f"  交易频率: {avg_trades_diff:+.0f}次\n")
            f.write(f"  最大回撤: {avg_max_dd_diff:+.2f}%\n")

        print(f"汇总报告已保存: {summary_path}")
        print("")


def _run_window_worker(validator, window, window_index, total_windows):
    """并行执行的Worker函数"""
    worker_start = time.time()
    window_id = window['window_id']
    start_date = window['start_date']
    end_date = window['end_date']

    print(f"  [{window_index}/{total_windows}] 窗口 {window_id}: "
          f"{start_date[:4]}-{start_date[4:6]} ~ {end_date[:4]}-{end_date[4:6]}")

    try:
        # 1. 运行实际回测
        actual_result = validator.run_atom_backtest(start_date, end_date)

        # 2. 计算理论结果
        theory_result = validator.calculate_theory_result(start_date, end_date)

        # 3. 对比指标
        comparison = validator.compare_metrics(actual_result, theory_result, window_id, start_date, end_date)

        worker_time = time.time() - worker_start
        print(f"      实际: 收益{actual_result['return_pct']:.2f}%, 夏普{actual_result['sharpe']:.2f}")
        print(f"      理论: 收益{theory_result['return_pct']:.2f}%, 夏普{theory_result['sharpe']:.2f}")
        print(f"      ⏱️  窗口总耗时: {worker_time:.1f}秒")

        return comparison

    except Exception as e:
        print(f"      错误: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='组合Atom滚动窗口验证器 - 实际执行 vs 理论测试')

    parser.add_argument('--atom', type=str, required=True, help='组合atom名称（如portfolio_rank3_combo）')
    parser.add_argument('--timeframe', type=str, default='d1', help='时间周期（默认: d1）')
    parser.add_argument('--window-months', type=int, default=12, help='窗口长度（月，默认: 12）')
    parser.add_argument('--step-months', type=int, default=12, help='滑动步长（月，默认: 12）')
    parser.add_argument('--workers', type=str, default='auto', help='并行进程数（默认: 1，可设置为auto）')
    parser.add_argument('--data-path', type=str, default='./data/nq_m1_forward_adjusted.csv', help='数据文件路径')
    parser.add_argument('--results-dir', type=str, default='backtest_results', help='回测结果目录')
    parser.add_argument('--output-dir', type=str, default='backtest_results/atom_validation', help='输出目录')

    args = parser.parse_args()

    # 解析workers参数
    if args.workers == 'auto':
        n_workers = mp.cpu_count()
    else:
        try:
            n_workers = int(args.workers)
        except:
            n_workers = 1

    # 创建验证器
    validator = RollingAtomValidator(
        atom_name=args.atom,
        timeframe=args.timeframe,
        window_months=args.window_months,
        step_months=args.step_months,
        data_path=args.data_path,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        n_workers=n_workers,
    )

    # 运行验证
    results = validator.run_validation(use_parallel=(n_workers > 1))

    # 生成报告
    validator.generate_report(results)


if __name__ == '__main__':
    main()
