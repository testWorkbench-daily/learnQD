#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Predictor - Jupyter Notebook 使用示例
在 Jupyter Notebook 或 T4 云环境中使用此代码
"""

# ============================================================
# 1. 导入必要的库
# ============================================================
import os
import sys
import torch
import pandas as pd
import numpy as np

# 如果 lstm_predictor.py 不在当前目录，添加路径
# sys.path.append('/path/to/lstm_predictor_dir')

# ============================================================
# 2. 配置参数（在这里修改你的设置）
# ============================================================
CONFIG = {
    'input': 'nq_m1_bars.csv',              # 输入CSV文件路径
    'lookback': 60,                          # 输入窗口长度
    'horizon': 5,                            # 预测时间跨度
    'output_dir': './lstm_output',           # 输出目录
    'epochs': 50,                            # 训练轮数（T4环境可以设小一点测试）
    'batch_size': 256,                       # 批次大小（CUDA可以设大一些，512或1024）
    'device': 'auto',                        # auto自动选择，或指定 cuda/mps/cpu
    
    # 时间范围过滤（可选）- 现在会真正生效了！
    'start': '2024-01-01',                   # 开始日期，默认为None使用全部数据
    'end': '2024-12-31',                     # 结束日期，默认为None使用全部数据
    
    # 其他可选参数
    # 'model_path': 'lstm_output/lstm_L60_H5.pt',  # 仅推理时指定
    # 'inference_only': True,                       # 仅推理模式
}

# ============================================================
# 3. 检查环境
# ============================================================
print("=" * 60)
print("环境检查")
print("=" * 60)
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"CPU 核心数: {os.cpu_count()}")
print("=" * 60 + "\n")

# ============================================================
# 4. 导入并运行 LSTM 预测器
# ============================================================
# 方法1: 直接导入并运行
from lstm_predictor import main

# 运行训练和预测
main(config=CONFIG)

# ============================================================
# 5. 查看结果（可选）
# ============================================================
# 读取生成的信号CSV
output_file = f"{CONFIG['output_dir']}/{CONFIG['input'].replace('.csv', '')}_signal_L{CONFIG['lookback']}_H{CONFIG['horizon']}.csv"
if os.path.exists(output_file):
    signals = pd.read_csv(output_file)
    print(f"\n生成的信号文件预览:")
    print(signals.head(10))
    print(f"\n信号统计:")
    print(signals['signal'].describe())
