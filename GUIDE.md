# NQ期货回测系统指南

## 需求说明

### 核心需求
- 使用NQ期货真实历史数据进行量化回测
- 支持多时间周期（m1/m5/m15/m30/h1/h4/d1）
- 自动生成交易记录CSV和可视化图表
- 模块化策略设计，通过Atom快速编写和测试策略

### 技术架构
```
bt_base.py      # 基类：StrategyAtom, BaseStrategy, Sizer, Analyzer
bt_runner.py    # 运行器：数据加载、回测执行、结果收集
atoms/          # 策略目录：每个策略一个文件
bt_main.py      # 主入口：命令行接口
```

### 设计原则
- **Atom模式**：每个策略是一个独立的Atom类，包含策略逻辑和仓位管理
- **Runner固定**：Runner处理通用逻辑（数据加载、broker配置、分析器），不需要修改
- **快速迭代**：新增策略只需创建新的Atom文件

## 使用说明

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --start | 开始日期 | 1900-01-01 |
| --end | 结束日期 | 当前日期 |
| --data | 数据文件 | nq_m1_all_backtrader.csv |
| --timeframe | K线周期 | d1 |
| --atom | 策略名称 | sma_cross |
| --compare | 多策略对比 | - |
| --no-save | 不保存CSV | 默认保存 |
| --no-plot | 不生成图表 | 默认生成 |
| --list | 列出可用策略 | - |

### 运行示例

```bash
# 默认运行
python bt_main.py

# 指定策略和时间
python bt_main.py --atom rsi_reversal --start 2024-01-01 --end 2024-12-31

# 使用1小时K线
python bt_main.py --timeframe h1

# 多策略对比
python bt_main.py --compare --timeframe d1 --start 2024-01-01 --end 2024-12-31

# 列出可用策略
python bt_main.py --list
```

### 创建新策略

```python
# atoms/my_strategy.py
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy

class MyStrategyAtom(StrategyAtom):
    name = "my_strategy"
    params = {'period': 14, 'threshold': 50}
    
    def strategy_cls(self):
        p = self.params
        
        class Strategy(BaseStrategy):
            params = (('period', p['period']), ('threshold', p['threshold']))
            
            def __init__(self):
                super().__init__()
                self.indicator = bt.ind.RSI(period=self.p.period)
            
            def next(self):
                if self.order:
                    return
                if not self.position:
                    if self.indicator < self.p.threshold:
                        self.order = self.buy()
                else:
                    if self.indicator > 100 - self.p.threshold:
                        self.order = self.sell()
        
        return Strategy
    
    def sizer_cls(self):
        # 可选：自定义仓位管理
        return None
```

注册策略（在bt_main.py的ATOMS字典中添加）:
```python
from atoms.my_strategy import MyStrategyAtom
ATOMS['my_strategy'] = MyStrategyAtom
```

## 文件说明

### 代码文件

| 文件 | 说明 |
|------|------|
| bt_base.py | 基类定义（StrategyAtom, BaseStrategy, Sizer, Analyzer） |
| bt_runner.py | 回测运行器（数据加载、执行、结果收集） |
| bt_main.py | 命令行入口 |
| plot_utils.py | 画图工具 |
| atoms/ | 策略目录 |

### atoms目录

| 文件 | 策略 |
|------|------|
| sma_cross.py | 双均线交叉（SMACrossAtom, SMACross_5_20, SMACross_10_30） |
| rsi_reversal.py | RSI反转（RSIReversalAtom, RSI_14_30_70） |
| macd_trend.py | MACD趋势（MACDTrendAtom, MACD_12_26_9） |
| bollinger_breakout.py | 布林带突破（BollingerBreakoutAtom, BollingerMeanReversion） |

### 数据文件

| 文件 | 说明 |
|------|------|
| nq_m1_all_backtrader.csv | NQ期货1分钟原始数据（357MB，690万行） |
| nq_adjusted.csv | 前复权后数据（消除换期跳空） |

### 数据处理工具

**forward_adjust.py** - NQ期货前复权处理

```bash
# 默认运行（输出到nq_adjusted.csv）
python forward_adjust.py

# 只检测缺口，不执行调整
python forward_adjust.py --dry-run

# 指定输入输出
python forward_adjust.py --input data.csv --output adjusted.csv
```

功能：
- 自动识别换期日（3/6/9/12月第三个周五）
- 清洗异常数据（删除负数和过低价格）
- 计算并消除换期跳空缺口
- 验证调整结果

### 输出文件

**backtest_results/**
- trades_{策略名}_{周期}_{时间戳}.csv

**CSV字段**
- trade_id, datetime, type, price, size, value, commission
- portfolio_value, cash, pnl
