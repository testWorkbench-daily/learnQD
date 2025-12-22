"""
Backtrader 可视化工具模块
提供回测结果的图表生成功能
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# 设置中文字体支持
try:
    # 尝试设置中文字体
    rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    pass


def plot_backtest_results(cerebro, timeframe='m1', save_path=None):
    """
    生成回测结果的可视化图表
    
    Args:
        cerebro: Backtrader的Cerebro对象（已运行完回测）
        timeframe: K线周期 ('m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1')
        save_path: 图表保存路径（可选），如果提供则保存图片而不显示
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    print('\n' + '=' * 60)
    print('📊 生成可视化图表')
    print('=' * 60)
    
    try:
        # 根据K线周期设置时间格式
        timeframe_config = {
            'm1': '%Y-%m-%d\n%H:%M',      # 分钟：显示日期和时间
            'm5': '%Y-%m-%d\n%H:%M',
            'm15': '%Y-%m-%d\n%H:%M',
            'm30': '%Y-%m-%d\n%H:%M',
            'h1': '%Y-%m-%d\n%H:00',      # 小时：显示日期和小时
            'h4': '%Y-%m-%d\n%H:00',
            'd1': '%Y-%m-%d',             # 日线：只显示日期
        }
        
        # 获取时间格式
        fmt_x_ticks = timeframe_config.get(timeframe, '%Y-%m-%d')
        
        # 设置图表样式参数
        plot_config = {
            'style': 'candlestick',         # K线图样式
            'barup': 'red',                 # 上涨K线颜色（中国习惯）
            'bardown': 'green',             # 下跌K线颜色（中国习惯）
            'volume': True,                 # 显示成交量
            'volup': 'red',                 # 上涨成交量颜色
            'voldown': 'green',             # 下跌成交量颜色
            'fmt_x_ticks': fmt_x_ticks,     # 时间格式
        }
        
        # 生成图表
        figs = cerebro.plot(**plot_config)
        
        # 如果提供了保存路径，保存图片
        if save_path:
            if figs and len(figs) > 0:
                fig = figs[0][0]  # 获取第一个图表
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f'✅ 图表已保存到: {save_path}')
                plt.close(fig)
            else:
                print('⚠️  无法获取图表对象')
                return False
        else:
            print('✅ 图表已生成并显示')
            print('提示: 关闭图表窗口以继续程序执行')
        
        return True
        
    except ImportError as e:
        print(f'⚠️  导入错误: {e}')
        print('提示: 请确保已安装 matplotlib')
        print('安装命令: pip install matplotlib')
        return False
        
    except Exception as e:
        print(f'⚠️  绘图失败: {e}')
        import traceback
        traceback.print_exc()
        return False


def plot_with_custom_style(cerebro, timeframe='m1', title=None, save_path=None):
    """
    使用自定义样式生成图表（更详细的配置）
    
    Args:
        cerebro: Backtrader的Cerebro对象
        timeframe: K线周期
        title: 图表标题（可选）
        save_path: 保存路径（可选）
        
    Returns:
        bool: 成功返回True，失败返回False
    """
    print('\n' + '=' * 60)
    print('📊 生成自定义样式图表')
    print('=' * 60)
    
    try:
        # 时间格式配置
        timeframe_config = {
            'm1': ('%Y-%m-%d\n%H:%M', '1分钟'),
            'm5': ('%Y-%m-%d\n%H:%M', '5分钟'),
            'm15': ('%Y-%m-%d\n%H:%M', '15分钟'),
            'm30': ('%Y-%m-%d\n%H:%M', '30分钟'),
            'h1': ('%Y-%m-%d\n%H:00', '1小时'),
            'h4': ('%Y-%m-%d\n%H:00', '4小时'),
            'd1': ('%Y-%m-%d', '日线'),
        }
        
        fmt_x_ticks, tf_name = timeframe_config.get(timeframe, ('%Y-%m-%d', timeframe))
        
        # 自定义图表配置
        plot_config = {
            'style': 'candlestick',
            'barup': 'red',
            'bardown': 'green',
            'volume': True,
            'volup': 'red',
            'voldown': 'green',
            'fmt_x_ticks': fmt_x_ticks,
            'plotdist': 0.1,              # 子图间距
            'plotabove': False,           # 主图位置
        }
        
        # 生成图表
        figs = cerebro.plot(**plot_config)
        
        # 添加标题
        if title and figs and len(figs) > 0:
            fig = figs[0][0]
            fig.suptitle(title or f'NQ期货回测结果 - {tf_name}K线', 
                        fontsize=14, fontweight='bold', y=0.995)
        
        # 保存或显示
        if save_path and figs and len(figs) > 0:
            fig = figs[0][0]
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'✅ 图表已保存到: {save_path}')
            plt.close(fig)
        else:
            print('✅ 图表已生成并显示')
        
        return True
        
    except Exception as e:
        print(f'⚠️  绘图失败: {e}')
        import traceback
        traceback.print_exc()
        return False


def plot_and_save(cerebro, timeframe='m1', output_dir='chart_output', 
                  filename_prefix='backtest_chart'):
    """
    生成图表并自动保存到指定目录
    
    Args:
        cerebro: Backtrader的Cerebro对象
        timeframe: K线周期
        output_dir: 输出目录
        filename_prefix: 文件名前缀
        
    Returns:
        str: 保存的文件路径，失败返回None
    """
    import os
    from datetime import datetime
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{filename_prefix}_{timeframe}_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
        
        # 生成并保存图表
        success = plot_backtest_results(cerebro, timeframe, save_path=filepath)
        
        if success:
            return filepath
        else:
            return None
            
    except Exception as e:
        print(f'⚠️  保存图表失败: {e}')
        return None


def check_plot_requirements():
    """
    检查绘图所需的依赖库是否已安装
    
    Returns:
        tuple: (bool, str) - (是否满足要求, 提示信息)
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return True, f"✅ matplotlib 已安装 (版本: {matplotlib.__version__})"
    except ImportError:
        return False, "❌ matplotlib 未安装\n安装命令: pip install matplotlib"


def get_available_styles():
    """
    获取可用的matplotlib样式列表
    
    Returns:
        list: 可用的样式名称列表
    """
    try:
        import matplotlib.pyplot as plt
        return plt.style.available
    except:
        return []


if __name__ == '__main__':
    """
    测试模块功能
    """
    print("=" * 60)
    print("Backtrader 可视化工具模块")
    print("=" * 60)
    
    # 检查依赖
    has_deps, msg = check_plot_requirements()
    print(f"\n{msg}")
    
    if has_deps:
        # 显示可用样式
        styles = get_available_styles()
        if styles:
            print(f"\n可用的matplotlib样式 ({len(styles)}个):")
            for i, style in enumerate(styles[:10], 1):
                print(f"  {i}. {style}")
            if len(styles) > 10:
                print(f"  ... 还有 {len(styles) - 10} 个")
    
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("""
1. 基本用法:
   from plot_utils import plot_backtest_results
   plot_backtest_results(cerebro, timeframe='h1')

2. 保存图表:
   plot_backtest_results(cerebro, timeframe='d1', save_path='my_chart.png')

3. 自定义样式:
   from plot_utils import plot_with_custom_style
   plot_with_custom_style(cerebro, timeframe='h4', title='我的策略')

4. 自动保存:
   from plot_utils import plot_and_save
   filepath = plot_and_save(cerebro, timeframe='m5', output_dir='charts')
    """)

