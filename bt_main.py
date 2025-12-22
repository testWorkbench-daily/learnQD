#!/usr/bin/env python
"""
NQ期货回测系统 - 主入口

用法:
    # 单策略回测
    python bt_main.py
    
    # 指定策略
    python bt_main.py --atom sma_cross

    # 多策略对比
    python bt_main.py --compare
    
    # 指定时间范围
    python bt_main.py --start 2024-01-01 --end 2024-12-31 --compare
"""
import argparse
import datetime
from bt_runner import Runner
from atoms import SMACrossAtom, RSIReversalAtom, MACDTrendAtom, BollingerBreakoutAtom
from atoms.sma_cross import SMACross_5_20, SMACross_10_30, SMACross_20_60


# 可用策略注册表
ATOMS = {
    'sma_cross': SMACrossAtom,
    'sma_5_20': SMACross_5_20,
    'sma_10_30': SMACross_10_30,
    'sma_20_60': SMACross_20_60,
    'rsi_reversal': RSIReversalAtom,
    'macd_trend': MACDTrendAtom,
    'boll_breakout': BollingerBreakoutAtom,
}


def main():
    parser = argparse.ArgumentParser(description='NQ期货回测系统')
    parser.add_argument('--start', default='1900-01-01', help='开始日期')
    parser.add_argument('--end', default=datetime.datetime.now().strftime('%Y-%m-%d'), help='结束日期')
    parser.add_argument('--data', default='/Users/hong/PycharmProjects/prepareQD/nq_m1_forward_adjusted.csv', help='数据文件')
    parser.add_argument('--timeframe', default='d1', choices=['m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1'], help='K线周期')
    parser.add_argument('--atom', default='sma_cross', choices=list(ATOMS.keys()), help='策略名称')
    parser.add_argument('--compare', action='store_true', help='多策略对比模式')
    parser.add_argument('--no-save', dest='save', action='store_false', help='不保存交易记录')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='不生成图表')
    parser.add_argument('--list', action='store_true', help='列出可用策略')
    parser.set_defaults(save=True, plot=True)
    
    args = parser.parse_args()
    
    # 列出策略
    if args.list:
        print('可用策略:')
        for name, cls in ATOMS.items():
            print(f'  {name:<15} - {cls().name}')
        return
    
    # 解析日期
    start = datetime.datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.datetime.strptime(args.end, '%Y-%m-%d')
    
    # 创建Runner
    runner = Runner(
        data_path=args.data,
        timeframe=args.timeframe,
        start_date=start,
        end_date=end,
    )
    
    if args.compare:
        # 多策略对比
        atoms = [cls() for cls in ATOMS.values()]
        runner.run_multiple(atoms, save_trades=False, plot=False)
    else:
        # 单策略回测
        atom_cls = ATOMS[args.atom]
        result = runner.run(atom_cls(), save_trades=args.save, plot=args.plot)


if __name__ == '__main__':
    main()

