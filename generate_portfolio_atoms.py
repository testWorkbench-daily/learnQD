#!/usr/bin/env python3
"""
根据稳健组合配置生成组合策略Atom代码

从export_robust_portfolios.py生成的CSV中读取组合配置，
自动生成可以在backtrader中运行的组合策略atom代码。

用法:
    python generate_portfolio_atoms.py --csv robust_portfolios_export_d1_top20.csv --top 5
    python generate_portfolio_atoms.py --csv robust_portfolios_export_d1_top20.csv --rank 1
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


def generate_atom_names(rank: int, prefix: str, rank_format: str) -> Tuple[str, str, str]:
    """
    生成Atom的各种名称

    Args:
        rank: 排名 (1, 2, 3, ...)
        prefix: 前缀 (如 "proj0113_")
        rank_format: 排名格式 ("" / "top" / "rank")

    Returns:
        (atom_name, class_name, filename)

    Examples:
        >>> generate_atom_names(1, "proj0113_", "top")
        ("proj0113_top0001", "Proj0113Top0001Atom", "proj0113_top0001.py")

        >>> generate_atom_names(5, "portfolio_robust_rank", "")
        ("portfolio_robust_rank5", "PortfolioRobustRank5Atom", "portfolio_robust_rank5.py")
    """
    if rank_format == 'top':
        rank_str = f"top{rank:04d}"
    elif rank_format == 'rank':
        rank_str = f"rank{rank:04d}"
    else:
        rank_str = str(rank)

    atom_name = f"{prefix}{rank_str}"

    # 生成类名：驼峰式 + Atom后缀
    # proj0113_top0001 → Proj0113Top0001Atom
    class_name = ''.join(word.capitalize() for word in atom_name.split('_')) + 'Atom'

    filename = f"{atom_name}.py"

    return atom_name, class_name, filename


def generate_portfolio_atom_code(
    atom_name: str,
    class_name: str,
    rank: int,
    strategies: List[str],
    weights: List[float],
    config_info: Dict
) -> str:
    """
    生成组合策略Atom代码

    Args:
        atom_name: Atom注册名 (如 "proj0113_top0001")
        class_name: 类名 (如 "Proj0113Top0001Atom")
        rank: 排名
        strategies: 策略列表
        weights: 权重列表
        config_info: 配置信息（夏普、收益等）

    Returns:
        Python代码字符串
    """
    # 使用传入的atom_name和class_name
    portfolio_name = atom_name

    # 生成策略列表字符串
    strategies_str = ', '.join([f'"{s}"' for s in strategies])

    # 生成权重字典字符串
    weights_dict_items = [f'"{s}": {w:.4f}' for s, w in zip(strategies, weights)]
    weights_dict_str = '{' + ', '.join(weights_dict_items) + '}'

    # 生成代码
    code = f'''"""
组合策略: 稳健排名 #{rank}

组成策略: {', '.join(strategies)}
权重配置: {', '.join([f'{w:.2%}' for w in weights])}

稳健性指标:
- 推荐频率: {config_info['recommend_freq']:.1%}
- 平均夏普: {config_info['avg_sharpe']:.3f}
- 夏普范围: {config_info['worst_sharpe']:.3f} ~ {config_info['best_sharpe']:.3f}
- 稳健评分: {config_info['robustness_score']:.3f}

最佳配置表现:
- 预期夏普: {config_info['config_expected_sharpe']:.3f}
- 预期收益: {config_info['config_expected_return']:.2%}
- 预期最大回撤: {config_info['config_expected_max_dd']:.2%}
"""

from bt_base import StrategyAtom, BaseStrategy
import backtrader as bt


class {class_name}(StrategyAtom):
    """稳健组合策略 - 排名#{rank}"""

    name = "{portfolio_name}"

    # 组合配置
    constituent_strategies = [{strategies_str}]
    weights = {weights_dict_str}

    def strategy_cls(self):
        """返回组合策略类"""
        constituent_strategies = self.constituent_strategies
        weights = self.weights

        class PortfolioStrategy(BaseStrategy):
            """
            组合策略实现

            采用信号加权方式:
            1. 收集各子策略的买卖信号
            2. 根据权重计算综合信号强度
            3. 当综合信号超过阈值时执行交易
            """

            def __init__(self):
                super().__init__()

                # 初始化各子策略的信号指标
                # 注意: 这里需要根据实际策略实现信号逻辑
                # 示例代码使用简化的均线信号

                # TODO: 根据constituent_strategies导入并初始化各子策略
                # 这里需要你根据实际策略来实现信号计算
                self.signals = {{}}

                # 示例: 如果是均线策略，可以这样初始化
                # self.signals['sma_cross'] = bt.ind.CrossOver(self.data.close, bt.ind.SMA(period=20))

            def next(self):
                if self.order:
                    return

                # 计算加权信号
                # TODO: 根据各策略的实际信号计算综合得分
                # 这里需要你根据实际策略来实现

                # 简化示例:
                # weighted_signal = sum(self.signals[s][0] * weights[s] for s in constituent_strategies)

                # if not self.position:
                #     if weighted_signal > 0.5:  # 买入阈值
                #         self.buy()
                # else:
                #     if weighted_signal < -0.5:  # 卖出阈值
                #         self.sell()

                # 实际使用时，建议直接运行portfolio_backtest.py来回测组合
                # 它会自动加载各子策略的daily_values并计算加权组合收益

                pass  # 占位符

        return PortfolioStrategy
'''

    return code


def register_to_bt_main(
    atom_name: str,
    class_name: str,
    module_path: str,
    main_file: str = 'bt_main.py'
) -> bool:
    """
    自动在bt_main.py中注册Atom

    策略:
    1. 在import区域添加: from atoms.portfolios.proj0113_top0001 import Proj0113Top0001Atom
    2. 在ATOMS字典中添加: 'proj0113_top0001': Proj0113Top0001Atom,
    3. 保持代码格式整齐

    Args:
        atom_name: Atom注册名 (如 "proj0113_top0001")
        class_name: 类名 (如 "Proj0113Top0001Atom")
        module_path: 模块路径 (如 "atoms.portfolios.proj0113_top0001")
        main_file: bt_main.py文件路径

    Returns:
        是否成功
    """
    # 读取文件
    with open(main_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检查是否已注册
    if any(f"'{atom_name}'" in line for line in lines):
        print(f"  ⚠️  警告: {atom_name} 已在 {main_file} 中注册，跳过")
        return False

    # 找到插入位置
    import_insert_idx = None
    atoms_start_idx = None
    atoms_end_idx = None

    for i, line in enumerate(lines):
        # 找到portfolio imports的位置（在其他imports之后）
        if 'from atoms.portfolios' in line and import_insert_idx is None:
            import_insert_idx = i

        # 找到ATOMS字典的开始和结束
        if 'ATOMS = {' in line:
            atoms_start_idx = i

        if atoms_start_idx is not None and '}' in line and atoms_end_idx is None:
            atoms_end_idx = i

    # 如果没有找到portfolio imports，在所有imports末尾添加
    if import_insert_idx is None:
        # 找到最后一个完整的import语句（跳过多行import）
        last_import_idx = None
        in_multiline = False
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                if '(' in line and ')' not in line:
                    # 开始多行import
                    in_multiline = True
                elif ')' in line:
                    # 结束多行import
                    in_multiline = False
                    last_import_idx = i
                elif not in_multiline:
                    # 单行import
                    last_import_idx = i

        if last_import_idx is not None:
            import_insert_idx = last_import_idx

    # 添加import
    if import_insert_idx is not None:
        # 如果有portfolio imports，找到最后一个
        last_portfolio_import = import_insert_idx
        if 'from atoms.portfolios' in lines[import_insert_idx]:
            for i in range(import_insert_idx, len(lines)):
                if 'from atoms.portfolios' in lines[i]:
                    last_portfolio_import = i
                elif lines[i].startswith('from ') or lines[i].startswith('import '):
                    break

        import_line = f"from {module_path} import {class_name}\n"
        lines.insert(last_portfolio_import + 1, import_line)
        print(f"  ✓ 添加import: from {module_path} import {class_name}")
    else:
        print(f"  ⚠️  警告: 未找到import插入位置")
        return False

    # 添加到ATOMS字典（在最后一个条目后）
    if atoms_end_idx is not None:
        # 在}之前添加新条目
        atoms_entry = f"    '{atom_name}': {class_name},\n"
        lines.insert(atoms_end_idx, atoms_entry)
        print(f"  ✓ 添加ATOMS条目: '{atom_name}': {class_name}")
    else:
        print(f"  ⚠️  警告: 未找到ATOMS字典")
        return False

    # 写回文件
    with open(main_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return True


def generate_atoms_from_csv(
    csv_file: str,
    output_dir: str = 'atoms',
    top_n: int = None,
    ranks: List[int] = None,
    prefix: str = 'portfolio_robust_rank',
    rank_format: str = '',
    register_to_main: bool = False,
    main_file: str = 'bt_main.py'
):
    """
    从CSV生成组合策略Atom代码

    Args:
        csv_file: 输入CSV文件
        output_dir: 输出目录
        top_n: 生成前N个组合
        ranks: 生成指定排名的组合
        prefix: Atom名称前缀 (默认: portfolio_robust_rank)
        rank_format: 排名格式 ("" / "top" / "rank")
        register_to_main: 是否自动注册到bt_main.py
        main_file: bt_main.py文件路径
    """
    # 读取CSV
    df = pd.read_csv(csv_file)
    print(f"读取了 {len(df)} 个稳健组合")

    # 确定要生成的组合
    if ranks:
        selected = df[df['rank'].isin(ranks)]
        print(f"选择了排名 {ranks} 的组合")
    elif top_n:
        selected = df.head(top_n)
        print(f"选择了前 {top_n} 个组合")
    else:
        selected = df
        print(f"将生成所有 {len(df)} 个组合")

    if len(selected) == 0:
        print("错误: 没有选择任何组合")
        return

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 生成代码
    print(f"\n开始生成Atom代码...")
    print("=" * 80)

    for idx, row in selected.iterrows():
        rank = int(row['rank'])
        num_strategies = int(row['num_strategies'])

        # 生成名称
        atom_name, class_name, filename = generate_atom_names(rank, prefix, rank_format)

        # 提取策略和权重
        strategies = []
        weights = []
        for i in range(1, num_strategies + 1):
            strategy = row.get(f'strategy_{i}')
            weight = row.get(f'weight_{i}')
            if pd.notna(strategy) and pd.notna(weight):
                strategies.append(strategy)
                weights.append(float(weight))

        # 配置信息
        config_info = {
            'recommend_freq': row['recommend_freq'],
            'avg_sharpe': row['avg_sharpe'],
            'worst_sharpe': row['worst_sharpe'],
            'best_sharpe': row['best_sharpe'],
            'robustness_score': row['robustness_score'],
            'config_expected_sharpe': row['config_expected_sharpe'],
            'config_expected_return': row['config_expected_return'],
            'config_expected_max_dd': row['config_expected_max_dd']
        }

        # 生成代码
        code = generate_portfolio_atom_code(atom_name, class_name, rank, strategies, weights, config_info)

        # 保存文件
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        print(f"✓ 排名 {rank}: {filename}")
        print(f"  Atom名: {atom_name}")
        print(f"  类名: {class_name}")
        print(f"  策略: {', '.join(strategies)}")
        print(f"  权重: {', '.join([f'{w:.2%}' for w in weights])}")

        # 自动注册
        if register_to_main:
            module_path = f"{output_dir.replace('/', '.')}.{atom_name}"
            register_to_bt_main(atom_name, class_name, module_path, main_file)

        print("")

    print("=" * 80)
    print(f"\n完成! 共生成 {len(selected)} 个文件")
    print(f"输出目录: {output_path.absolute()}")
    print(f"\n注意: 生成的代码是模板，需要根据实际策略实现信号逻辑")
    print(f"建议: 使用 portfolio_backtest.py 直接回测组合（不需要修改代码）")


def main():
    parser = argparse.ArgumentParser(description='根据稳健组合配置生成Atom代码')
    parser.add_argument('--csv', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--output-dir', type=str, default='atoms/portfolios',
                        help='输出目录 (默认: atoms/portfolios)')
    parser.add_argument('--top', type=int, default=None,
                        help='生成前N个组合')
    parser.add_argument('--ranks', type=int, nargs='+', default=None,
                        help='生成指定排名的组合 (如: --ranks 1 2 3)')
    parser.add_argument('--prefix', type=str, default='portfolio_robust_rank',
                        help='生成的atom名称前缀 (默认: portfolio_robust_rank)')
    parser.add_argument('--rank-format', type=str, default='',
                        choices=['', 'top', 'rank'],
                        help='排名格式: "" → rank1, "top" → top0001, "rank" → rank0001')
    parser.add_argument('--register-to-main', action='store_true',
                        help='自动注册到bt_main.py的ATOMS字典')
    parser.add_argument('--main-file', type=str, default='bt_main.py',
                        help='bt_main.py文件路径 (默认: bt_main.py)')

    args = parser.parse_args()

    print("=" * 80)
    print("组合策略Atom代码生成器")
    print("=" * 80)
    print(f"输入文件: {args.csv}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    print()

    try:
        generate_atoms_from_csv(
            csv_file=args.csv,
            output_dir=args.output_dir,
            top_n=args.top,
            ranks=args.ranks,
            prefix=args.prefix,
            rank_format=args.rank_format,
            register_to_main=args.register_to_main,
            main_file=args.main_file
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
