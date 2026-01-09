#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM Price Direction Predictor
根据PRD_LSTM_Predictor.md实现的LSTM预测信号生成器
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os


class FeatureEngine:
    """特征工程类 - 生成8个技术指标特征"""
    
    def __init__(self):
        self.feature_names = [
            'returns', 'log_returns', 'high_low_pct', 'close_position',
            'volume_change', 'returns_ma5', 'returns_ma20', 'volatility'
        ]
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        计算8个特征
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            features: shape [N, 8]
        """
        df = df.copy()
        
        # 1. returns
        df['returns'] = df['close'].pct_change()
        
        # 2. log_returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 3. high_low_pct (振幅)
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        
        # 4. close_position (K线位置)
        hl_range = df['high'] - df['low']
        df['close_position'] = (df['close'] - df['low']) / (hl_range + 1e-8)
        
        # 5. volume_change (量比)
        volume_ma20 = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_change'] = df['volume'] / (volume_ma20 + 1e-8) - 1
        
        # 6. returns_ma5 (短期动量)
        df['returns_ma5'] = df['returns'].rolling(window=5, min_periods=1).mean()
        
        # 7. returns_ma20 (中期动量)
        df['returns_ma20'] = df['returns'].rolling(window=20, min_periods=1).mean()
        
        # 8. volatility (波动率)
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
        
        # 提取特征矩阵
        features = df[self.feature_names].values
        
        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features


class TargetEngine:
    """标签构造类"""
    
    def __init__(self, horizon: int = 10, scale: float = 0.01):
        """
        Args:
            horizon: 预测未来多少个bar
            scale: tanh缩放参数，默认0.005 (0.5%)
        """
        self.horizon = horizon
        self.scale = scale
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        构造标签: tanh(future_return / scale)
        
        Args:
            df: 包含close列的DataFrame
            
        Returns:
            labels: shape [N], 值域[-1, +1]
        """
        close = df['close'].values
        
        # 未来收益率
        future_close = np.roll(close, -self.horizon)
        future_return = (future_close - close) / (close + 1e-8)
        
        # tanh映射到[-1, +1]
        labels = np.tanh(future_return / self.scale)
        
        # 最后horizon个样本没有标签，设为0
        labels[-self.horizon:] = 0.0
        
        return labels


class TimeSeriesDataset(Dataset):
    """时序数据集"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, lookback: int):
        """
        Args:
            features: [N, F] 特征矩阵
            labels: [N] 标签
            lookback: 输入窗口长度
        """
        self.features = features
        self.labels = labels
        self.lookback = lookback
    
    def __len__(self):
        return len(self.features) - self.lookback + 1
    
    def __getitem__(self, idx):
        # 取lookback个时间步
        x = self.features[idx:idx + self.lookback]
        y = self.labels[idx + self.lookback - 1]
        
        return torch.FloatTensor(x), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    """LSTM预测模型"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 32, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(input_size)
        
        # LSTM (单向)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, lookback, input_size]
            
        Returns:
            output: [batch_size, 1], 值域[-1, +1]
        """
        # LayerNorm
        x = self.layer_norm(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 取最后时间步的隐状态
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # 全连接层
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.tanh(out)
        
        return out


class Trainer:
    """训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, lr: float = 0.001, weight_decay: float = 0.0001,
              patience: int = 10) -> dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            lr: 学习率
            weight_decay: L2正则化系数
            patience: 早停耐心值
            
        Returns:
            training_log: 训练日志
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print(f"\n{'='*50}")
        print(f"开始训练 | Epochs: {epochs} | LR: {lr}")
        print(f"{'='*50}\n")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\n早停触发！在epoch {epoch+1}停止训练")
                break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        training_log = {
            'epochs_trained': epoch + 1,
            'best_val_loss': best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        print(f"\n{'='*50}")
        print(f"训练完成 | 最佳验证Loss: {best_val_loss:.6f}")
        print(f"{'='*50}\n")
        
        return training_log


class Predictor:
    """预测器 - 生成信号CSV"""
    
    def __init__(self, model: nn.Module, scaler: StandardScaler, 
                 lookback: int, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.lookback = lookback
        self.device = device
    
    def predict(self, df: pd.DataFrame, feature_engine: FeatureEngine, batch_size: int = 256) -> pd.DataFrame:
        """
        生成预测信号（批处理模式，提高并发性能）
        
        Args:
            df: 原始OHLCV数据
            feature_engine: 特征工程对象
            batch_size: 批处理大小
            
        Returns:
            signal_df: 包含ts_event和signal的DataFrame
        """
        # 计算特征
        features = feature_engine.transform(df)
        
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 初始化信号数组
        signals = np.full(len(features_scaled), np.nan)
        
        # 收集所有有效样本的索引
        valid_indices = list(range(self.lookback - 1, len(features_scaled)))
        
        print("\n生成预测信号中...")
        
        # 批处理推理
        for batch_start in tqdm(range(0, len(valid_indices), batch_size)):
            batch_end = min(batch_start + batch_size, len(valid_indices))
            batch_idx = valid_indices[batch_start:batch_end]
            
            # 构建批样本
            batch_windows = []
            for i in batch_idx:
                window = features_scaled[i - self.lookback + 1:i + 1]
                batch_windows.append(window)
            
            # 转为张量
            x_batch = torch.FloatTensor(np.array(batch_windows)).to(self.device)
            
            # 批推理
            with torch.no_grad():
                preds = self.model(x_batch)
                pred_values = preds.cpu().numpy().flatten()
            
            # 填充结果
            for j, idx in enumerate(batch_idx):
                signals[idx] = pred_values[j]
        
        # 构造输出DataFrame
        signal_df = pd.DataFrame({
            'ts_event': df['ts_event'],
            'signal': signals
        })
        
        # 保留3位小数
        signal_df['signal'] = signal_df['signal'].round(3)
        
        return signal_df


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    评估预测效果
    
    Args:
        y_true: 真实标签
        y_pred: 预测值
        
    Returns:
        metrics: 评估指标字典
    """
    # 移除NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # IC (Information Coefficient)
    ic = np.corrcoef(y_true, y_pred)[0, 1]
    
    # 方向准确率
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    direction_acc = np.mean(direction_true == direction_pred)
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2)
    
    return {
        'IC': ic,
        'direction_accuracy': direction_acc,
        'MSE': mse
    }


def main(config=None):
    """
    主函数
    
    Args:
        config: dict, 可选的配置字典。如果为None，则使用命令行参数
    """
    if config is None:
        # 命令行模式
        parser = argparse.ArgumentParser(
            description='LSTM价格方向预测器',
            epilog='示例: python lstm_predictor.py --input data.csv --device auto --start 2024-01-01 --end 2024-12-31'
        )
        parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
        parser.add_argument('--lookback', type=int, default=60, help='输入窗口长度')
        parser.add_argument('--horizon', type=int, default=5, help='预测时间跨度')
        parser.add_argument('--output_dir', type=str, default='./lstm_output', help='输出目录')
        parser.add_argument('--model_path', type=str, default=None, help='已训练模型路径（仅推理时使用）')
        parser.add_argument('--inference_only', action='store_true', help='仅推理模式')
        parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
        parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
        parser.add_argument('--device', type=str, default='auto', 
                            help='设备选择: auto (自动), cuda, mps, cpu. auto会按CUDA→MPS→CPU顺序尝试')
        parser.add_argument('--start', type=str, default=None, help='开始日期（格式: YYYY-MM-DD），默认为数据起始日期')
        parser.add_argument('--end', type=str, default=None, help='结束日期（格式: YYYY-MM-DD），默认为数据结束日期')
        
        args = parser.parse_args()
    else:
        # 字典配置模式（Jupyter Notebook）
        class DictToArgs:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    setattr(self, key, value)
        
        # 设置默认值
        default_config = {
            'input': None,
            'lookback': 60,
            'horizon': 5,
            'output_dir': './lstm_output',
            'model_path': None,
            'inference_only': False,
            'epochs': 100,
            'batch_size': 64,
            'device': 'auto',
            'start': None,
            'end': None
        }
        default_config.update(config)
        args = DictToArgs(default_config)
        
        if args.input is None:
            raise ValueError("必须指定 'input' 参数")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备 - 智能选择：CUDA → MPS → CPU
    if args.device == 'auto':
        # 自动选择最优设备
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ 自动选择: CUDA GPU 加速 (检测到 NVIDIA GPU)")
            # 显示 CUDA 设备信息
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"✓ 自动选择: MPS GPU 加速 (检测到 Apple Silicon)")
        else:
            device = 'cpu'
            print(f"ℹ 自动选择: CPU 计算 (未检测到可用GPU)")
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ 使用 CUDA GPU 加速")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"⚠ CUDA 不可用，尝试 MPS...")
            if torch.backends.mps.is_available():
                device = 'mps'
                print(f"✓ 降级到 MPS GPU")
            else:
                device = 'cpu'
                print(f"⚠ MPS 也不可用，使用 CPU")
    elif args.device == 'mps':
        if torch.backends.mps.is_available():
            device = 'mps'
            print(f"✓ 使用 MPS GPU 加速")
        else:
            print(f"⚠ MPS 不可用，使用 CPU")
            device = 'cpu'
    else:
        device = 'cpu'
        print(f"ℹ 使用 CPU 计算")
    
    print(f"\n最终设备: {device}")
    if device == 'cpu':
        print(f"CPU 线程数: {torch.get_num_threads()}")
    
    # ========== 1. 加载数据 ==========
    print(f"\n加载数据: {args.input}")
    df = pd.read_csv(args.input)
    
    # 检查必需的列
    required_cols = ['ts_event', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    print(f"原始数据行数: {len(df)}")
    print(f"原始时间范围: {df['ts_event'].iloc[0]} ~ {df['ts_event'].iloc[-1]}")
    
    # ========== 时间范围过滤 ==========
    if args.start is not None or args.end is not None:
        print(f"\n应用时间范围过滤...")
        # 转换ts_event为datetime
        df['ts_event_dt'] = pd.to_datetime(df['ts_event'])
        
        if args.start is not None:
            start_dt = pd.to_datetime(args.start)
            df = df[df['ts_event_dt'] >= start_dt]
            print(f"  起始时间: {args.start}")
        
        if args.end is not None:
            end_dt = pd.to_datetime(args.end)
            df = df[df['ts_event_dt'] <= end_dt]
            print(f"  结束时间: {args.end}")
        
        # 删除临时列
        df = df.drop(columns=['ts_event_dt'])
        
        # 重置索引
        df = df.reset_index(drop=True)
        
        print(f"\n过滤后数据行数: {len(df)}")
        print(f"过滤后时间范围: {df['ts_event'].iloc[0]} ~ {df['ts_event'].iloc[-1]}")
        
        if len(df) < args.lookback + args.horizon + 100:
            raise ValueError(f"过滤后数据量太少（{len(df)}行），至少需要 {args.lookback + args.horizon + 100} 行")
    else:
        print(f"\n未指定时间范围，使用全部数据")
    
    # ========== 2. 特征工程 ==========
    print("\n计算特征...")
    feature_engine = FeatureEngine()
    features = feature_engine.transform(df)
    print(f"特征维度: {features.shape}")
    
    # ========== 3. 构造标签 ==========
    print("\n构造标签...")
    target_engine = TargetEngine(horizon=args.horizon)
    labels = target_engine.create_labels(df)
    print(f"标签维度: {labels.shape}")
    
    # ========== 4. 数据划分 ==========
    n = len(features)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = features[:train_end], labels[:train_end]
    X_val, y_val = features[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = features[val_end:], labels[val_end:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # ========== 5. 标准化 ==========
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 保存scaler
    scaler_path = output_dir / f"scaler_L{args.lookback}_H{args.horizon}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nScaler已保存: {scaler_path}")
    
    # ========== 6. 训练或加载模型 ==========
    # 根据设备选择 num_workers（GPU 使用 0，CPU 使用 4）
    num_workers = 0 if device in ['mps', 'cuda'] else 4
    
    if not args.inference_only:
        # 创建数据集
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train, args.lookback)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val, args.lookback)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        
        # 创建模型
        model = LSTMModel(input_size=8, hidden_size=32, num_layers=2, dropout=0.2)
        
        # 训练
        trainer = Trainer(model, device=device)
        training_log = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=0.001,
            weight_decay=0.0001,
            patience=10
        )
        
        # 保存模型
        model_path = output_dir / f"lstm_L{args.lookback}_H{args.horizon}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存: {model_path}")
        
        # 保存训练日志
        log_path = output_dir / f"training_log_L{args.lookback}_H{args.horizon}.json"
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        print(f"训练日志已保存: {log_path}")
        
        # 测试集评估
        print("\n" + "="*50)
        print("测试集评估")
        print("="*50)
        
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test, args.lookback)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        
        model.eval()
        y_test_pred = []
        y_test_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                y_test_pred.extend(outputs.cpu().numpy().flatten())
                y_test_true.extend(batch_y.numpy().flatten())
        
        metrics = evaluate_predictions(np.array(y_test_true), np.array(y_test_pred))
        print(f"IC (Information Coefficient): {metrics['IC']:.4f}")
        print(f"方向准确率: {metrics['direction_accuracy']*100:.1f}%")
        print(f"MSE: {metrics['MSE']:.4f}")
        print("="*50 + "\n")
        
    else:
        # 加载已有模型
        if args.model_path is None:
            raise ValueError("推理模式需要指定--model_path")
        
        model = LSTMModel(input_size=8, hidden_size=32, num_layers=2, dropout=0.2)
        # 智能加载模型（处理跨设备情况）
        if device == 'cuda':
            model.load_state_dict(torch.load(args.model_path, map_location='cuda'))
        elif device == 'mps':
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print(f"模型已加载: {args.model_path}")
    
    # ========== 7. 生成信号CSV ==========
    # 注意：对整个过滤后的数据集生成信号
    print(f"\n生成信号（数据量: {len(df)} 行）...")
    features_full = feature_engine.transform(df)
    scaler_full = StandardScaler()
    scaler_full.fit(features_full[:train_end])  # 只用训练集fit
    
    predictor = Predictor(model, scaler_full, args.lookback, device=device)
    signal_df = predictor.predict(df, feature_engine, batch_size=args.batch_size)
    
    # 保存信号CSV
    input_name = Path(args.input).stem
    output_name = f"{input_name}_signal_L{args.lookback}_H{args.horizon}.csv"
    output_path = output_dir / output_name
    signal_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print(f"信号CSV已生成: {output_path}")
    print(f"总行数: {len(signal_df)}")
    print(f"有效信号数: {signal_df['signal'].notna().sum()}")
    print("="*50)
    
    # 显示信号统计
    valid_signals = signal_df['signal'].dropna()
    if len(valid_signals) > 0:
        print(f"\n信号统计:")
        print(f"  均值: {valid_signals.mean():.4f}")
        print(f"  标准差: {valid_signals.std():.4f}")
        print(f"  最小值: {valid_signals.min():.4f}")
        print(f"  最大值: {valid_signals.max():.4f}")
        print(f"  看多信号(>0.5): {(valid_signals > 0.5).sum()} ({(valid_signals > 0.5).sum()/len(valid_signals)*100:.1f}%)")
        print(f"  看空信号(<-0.5): {(valid_signals < -0.5).sum()} ({(valid_signals < -0.5).sum()/len(valid_signals)*100:.1f}%)")


if __name__ == '__main__':
    # ====================================================================
    # Jupyter Notebook 配置模式（取消注释以启用）
    # ====================================================================
    # CONFIG = {
    #     'input': 'nq_m1_bars.csv',           # 输入CSV文件路径
    #     'lookback': 60,                       # 输入窗口长度
    #     'horizon': 5,                         # 预测时间跨度
    #     'output_dir': './lstm_output',        # 输出目录
    #     'epochs': 100,                        # 训练轮数
    #     'batch_size': 128,                    # 批次大小（GPU可以设大一些）
    #     'device': 'auto',                     # auto/cuda/mps/cpu
    #     # 'start': '2024-01-01',              # 可选：开始日期
    #     # 'end': '2024-12-31',                # 可选：结束日期
    #     # 'model_path': 'lstm_output/lstm_L60_H5.pt',  # 仅推理时指定
    #     # 'inference_only': True,               # 仅推理模式
    # }
    # ====================================================================
    
    # 打印设备信息
    print("\n" + "="*60)
    print("LSTM 价格方向预测器")
    print("="*60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"\n可用设备:")
    print(f"  CUDA: {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"  MPS:  {torch.backends.mps.is_available()} (内置: {torch.backends.mps.is_built()})")
    print(f"  CPU:  {os.cpu_count()} 核心")
    print("="*60 + "\n")
    
    # 检查是否使用配置字典模式
    if 'CONFIG' in locals() or 'CONFIG' in globals():
        print("📝 使用配置字典模式 (Jupyter Notebook)")
        main(config=CONFIG)
    else:
        print("📝 使用命令行参数模式")
        main()
