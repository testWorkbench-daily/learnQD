# NQ期货量化回测系统

## 📚 项目文档导航

### 🚀 快速开始
- **[QUICK_START.md](QUICK_START.md)** - 5分钟快速上手指南
- **[USAGE.md](USAGE.md)** ⭐ - 详细使用说明（推荐）

### 📖 深入了解
- **[FEATURES.md](FEATURES.md)** - 功能特性详解
- **[CSV_FILES_GUIDE.md](CSV_FILES_GUIDE.md)** - 交易记录CSV文件说明
- **[README_backtrader.md](README_backtrader.md)** - 完整技术文档

### 🔧 开发文档
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - 代码重构说明
- **plot_utils.py** - 画图工具模块源码

## 📁 核心文件

### 程序文件
```
backtrader_demo.py          # 主程序（907行）
plot_utils.py               # 画图模块（274行）
```

### 数据文件
```
nq_m1_cleaned.csv          # 清洗后的NQ期货分钟数据（210万行）
nq_m1_all_backtrader.csv   # 原始数据（包含脏数据，不推荐使用）
```

### 输出目录
```
backtest_results/          # 交易记录CSV文件
chart_output/              # 图表输出（如使用保存功能）
```

## ⚡ 快速使用

### 安装依赖
```bash
pip install backtrader pandas matplotlib
```

### 运行回测
```bash
# 最简单的用法
python backtrader_demo.py

# 使用小时线
python backtrader_demo.py --timeframe h1

# 保存交易记录
python backtrader_demo.py --timeframe h1 --save-trades

# 生成可视化图表
python backtrader_demo.py --timeframe d1 --plot

# 完整功能
python backtrader_demo.py --timeframe h4 --start 2024-01-01 --end 2024-06-30 --save-trades --plot
```

### 查看帮助
```bash
python backtrader_demo.py --help
```

## 🎯 核心功能

### 1. 多时间周期回测
- ✅ 支持7种K线周期（m1/m5/m15/m30/h1/h4/d1）
- ✅ 自动数据重采样
- ✅ 自适应策略参数

### 2. 交易记录导出
- ✅ 详细记录每笔交易（18个字段）
- ✅ 包含技术指标、账户状态、盈亏分析
- ✅ CSV格式，便于分析

### 3. 可视化图表
- ✅ K线图（红涨绿跌）
- ✅ 技术指标（SMA、RSI、MACD、布林带）
- ✅ 买卖信号标记
- ✅ 账户价值和回撤曲线

### 4. 灵活配置
- ✅ 命令行参数控制
- ✅ 自定义时间范围
- ✅ 选择性功能启用

## 📊 K线周期说明

| 参数 | 周期 | 适用场景 | 推荐时间范围 |
|------|------|----------|--------------|
| m1 | 1分钟 | 高频交易 | 1-7天 |
| m5 | 5分钟 | 日内短线 | 1周-1个月 |
| m15 | 15分钟 | 日内波段 | 1-2个月 |
| m30 | 30分钟 | 日内波段 | 1-3个月 |
| h1 | 1小时 | 短期趋势 | 1-6个月 |
| h4 | 4小时 | 中期趋势 | 3-12个月 |
| d1 | 日线 | 长期趋势 | 1-5年 |

## 💡 使用示例

### 场景1：快速验证策略
```bash
# 测试最近1个月的日线表现
python backtrader_demo.py --timeframe d1 --start 2024-11-01 --end 2024-12-31
```

### 场景2：详细分析
```bash
# 测试2024年上半年，保存所有交易记录
python backtrader_demo.py --timeframe h1 --start 2024-01-01 --end 2024-06-30 --save-trades
```

### 场景3：可视化展示
```bash
# 生成完整图表报告
python backtrader_demo.py --timeframe d1 --start 2024-01-01 --end 2024-12-31 --plot
```

### 场景4：完整功能测试
```bash
# 保存记录 + 生成图表
python backtrader_demo.py --timeframe h4 --start 2024-01-01 --end 2024-06-30 --save-trades --plot
```

## 🔍 输出说明

### 终端输出
- 回测进度和状态
- 最终资金和收益率
- 交易统计（次数、胜率、盈亏）
- 风险指标（最大回撤、夏普比率）

### CSV文件（--save-trades）
保存在 `backtest_results/` 目录：
- 每笔交易的完整信息
- 交易时的技术指标状态
- 账户价值变化
- 详细盈亏分析

### 图表（--plot）
包含：
- K线图和成交量
- 技术指标曲线
- 买卖信号标记
- 资金和回撤曲线

## 📈 数据说明

### 数据来源
- NQ期货（纳斯达克100指数期货）
- 分钟级历史数据
- 时间范围：2020-01-01 至 2025-12-19

### 数据质量
- ✅ 已清洗（删除异常值）
- ✅ 已去重（合并重复时间戳）
- ✅ 已验证（无缺失值）
- ✅ 共210万行数据

## 🛠️ 系统要求

### 软件环境
- Python 3.7+
- backtrader
- pandas
- matplotlib（用于图表）

### 硬件建议
- 内存：8GB+
- 存储：5GB+（用于数据文件）
- CPU：多核处理器（用于大数据量回测）

## ⚠️ 注意事项

1. **数据文件**：确保 `nq_m1_cleaned.csv` 存在
2. **时间范围**：使用较小K线周期时，建议缩短测试时间
3. **内存占用**：1分钟K线回测可能需要2GB+内存
4. **回测结果**：历史表现不代表未来收益

## 🤝 技术支持

### 查看文档
- 使用说明：`cat USAGE.md`
- 功能特性：`cat FEATURES.md`
- CSV指南：`cat CSV_FILES_GUIDE.md`

### 在线帮助
```bash
python backtrader_demo.py --help
python plot_utils.py  # 查看画图模块说明
```

### 常见问题
请查看 [USAGE.md](USAGE.md) 的"常见问题"部分

## 📝 更新日志

### v2.1 (2024-12-21)
- ✅ 重构：画图逻辑独立成模块
- ✅ 新增：详细使用文档（USAGE.md）
- ✅ 优化：简化主程序输出
- ✅ 完善：文档体系

### v2.0 (2024-12-21)
- ✅ 新增：多时间周期支持
- ✅ 新增：交易记录导出
- ✅ 新增：可视化图表生成
- ✅ 优化：命令行参数控制
- ✅ 删除：所有模拟数据逻辑

### v1.0
- ✅ 基础回测功能
- ✅ 双均线策略
- ✅ 数据清洗脚本

## 📄 许可证

MIT License

## 👨‍💻 作者

AI Assistant

---

**最后更新**: 2024-12-21  
**版本**: v2.1  
**状态**: ✅ 稳定版本

## 🎯 下一步

1. 查看 [QUICK_START.md](QUICK_START.md) 快速上手
2. 阅读 [USAGE.md](USAGE.md) 了解详细用法
3. 运行 `python backtrader_demo.py --help` 查看参数
4. 开始你的第一次回测！

**Happy Trading! 📈**

