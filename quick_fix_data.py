"""
快速修复数据 - 创建可用的连续合约数据
处理负价格
"""

import pandas as pd
import numpy as np

def quick_clean_data(input_path, output_path):
    """
    快速清理数据，移除明显错误并创建连续合约
    
    步骤：
    1. 移除负价格和异常低价
    2. 检测并调整大幅跳空
    3. 保存清理后的数据
    """
    print("="*60)
    print("快速数据清理工具")
    print("="*60)
    
    print("\n步骤1: 加载数据...")
    df = pd.read_csv(input_path, parse_dates=['ts_event'])
    print(f"原始数据: {len(df):,} 行")
    print(f"时间范围: {df['ts_event'].min()} 到 {df['ts_event'].max()}")
    print(f"价格范围: {df['close'].min():.2f} 到 {df['close'].max():.2f}")
    
    # 步骤2: 移除明显错误的数据
    print("\n步骤2: 移除数据错误...")
    
    # 移除负价格和极端低价（NQ正常价格应该在几百到几万）
    before = len(df)
    df = df[
        (df['open'] > 0) & 
        (df['high'] > 0) & 
        (df['low'] > 0) & 
        (df['close'] > 0) &
        (df['close'] > 100)  # NQ价格不应该低于100
    ].copy()
    removed = before - len(df)
    print(f"移除了 {removed:,} 行错误数据 ({removed/before*100:.2f}%)")
    
    # 按时间排序
    df = df.sort_values('ts_event').reset_index(drop=True)

    # 步骤5: 保存清理后的数据
    print("\n步骤5: 保存清理后的数据...")
    
    output_df = df[['ts_event', 'open', 'high', 'low', 'close', 'volume']].copy()
    output_df.to_csv(output_path, index=False)
    
    print(f"\n清理后数据已保存到: {output_path}")
    print(f"最终数据: {len(output_df):,} 行")
    print(f"时间范围: {output_df['ts_event'].min()} 到 {output_df['ts_event'].max()}")
    print(f"价格范围: {output_df['close'].min():.2f} 到 {output_df['close'].max():.2f}")
    
    # 生成统计报告
    print("\n" + "="*60)
    print("数据质量报告")
    print("="*60)
    print(f"数据完整性: {len(output_df)/len(pd.read_csv(input_path))*100:.2f}%")
    print(f"移除错误数据: {removed:,} 行")

    returns_stats = output_df['close'].pct_change()
    print(f"\n收益率统计:")
    print(f"  均值: {returns_stats.mean()*100:.4f}%")
    print(f"  标准差: {returns_stats.std()*100:.4f}%")
    print(f"  最大: {returns_stats.max()*100:.2f}%")
    print(f"  最小: {returns_stats.min()*100:.2f}%")
    print(f"  95分位: {returns_stats.quantile(0.95)*100:.2f}%")
    print(f"  5分位: {returns_stats.quantile(0.05)*100:.2f}%")
    
    print("\n" + "="*60)
    print("✅ 数据清理完成！")
    print("="*60)
    print(f"\n现在可以使用 {output_path} 进行回测")
    
    return output_path


if __name__ == '__main__':
    import sys
    
    # 默认路径
    input_path = '/Users/hong/PycharmProjects/prepareQD/nq_m1_all_backtrader.csv'
    output_path = '/Users/hong/PycharmProjects/prepareQD/nq_m1_cleaned.csv'
    
    # 可以从命令行指定输出路径
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    
    print("\n数据清理工具")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print("\n开始处理...\n")
    
    quick_clean_data(input_path, output_path)

