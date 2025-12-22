"""
策略原子基类和基础组件
"""
import backtrader as bt
from abc import ABC, abstractmethod
from typing import Type, Optional, Dict, Any, List


class StrategyAtom(ABC):
    """
    策略原子基类
    
    继承此类来实现一个完整的交易策略，包括：
    - 策略逻辑 (Strategy)
    - 仓位管理 (Sizer)
    - 自定义指标 (Indicators)
    """
    
    name: str = "unnamed"
    params: Dict[str, Any] = {}
    
    @abstractmethod
    def strategy_cls(self) -> Type[bt.Strategy]:
        """返回策略类"""
        pass
    
    def sizer_cls(self) -> Optional[Type[bt.Sizer]]:
        """返回Sizer类，默认None使用系统默认"""
        return None
    
    def indicators(self) -> List[Type[bt.Indicator]]:
        """返回自定义指标类列表"""
        return []
    
    def analyzers(self) -> List[Type[bt.Analyzer]]:
        """返回自定义分析器类列表"""
        return []


# =============================================================================
# 基础Sizer
# =============================================================================
class PercentSizer(bt.Sizer):
    """按账户百分比下单"""
    params = (('pct', 0.1),)
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return int(cash * self.p.pct / data.close[0])
        return self.broker.getposition(data).size


class FixedRiskSizer(bt.Sizer):
    """固定风险仓位管理"""
    params = (
        ('risk_pct', 0.02),
        ('stop_pct', 0.05),
    )
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            risk_amount = self.broker.getvalue() * self.p.risk_pct
            stop_distance = data.close[0] * self.p.stop_pct
            return max(int(risk_amount / stop_distance), 1)
        return self.broker.getposition(data).size


# =============================================================================
# 基础Analyzer
# =============================================================================
class TradeRecorder(bt.Analyzer):
    """交易记录器"""
    
    def __init__(self):
        self.trades = []
        self.trade_count = 0
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_count += 1
            self.trades.append({
                'id': self.trade_count,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'barlen': trade.barlen,
                'size': trade.size,
                'price': trade.price,
            })
    
    def get_analysis(self):
        if not self.trades:
            return {'total': 0, 'trades': []}
        
        wins = [t for t in self.trades if t['pnlcomm'] > 0]
        losses = [t for t in self.trades if t['pnlcomm'] <= 0]
        
        return {
            'total': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades) * 100,
            'total_pnl': sum(t['pnlcomm'] for t in self.trades),
            'avg_pnl': sum(t['pnlcomm'] for t in self.trades) / len(self.trades),
            'trades': self.trades,
        }


# =============================================================================
# 基础Strategy (带交易记录功能)
# =============================================================================
class BaseStrategy(bt.Strategy):
    """
    基础策略类，提供通用功能：
    - 交易记录
    - 日志输出
    - 常用指标
    """
    params = (
        ('printlog', False),
    )
    
    def __init__(self):
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.trade_records = []
        self.trade_count = 0
    
    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'[{dt}] {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status == order.Completed:
            dt = self.datas[0].datetime.datetime(0)
            
            if order.isbuy():
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                self._record_trade(dt, 'BUY', order)
            else:
                pnl = 0
                if self.buyprice:
                    pnl = (order.executed.price - self.buyprice) * order.executed.size - order.executed.comm - (self.buycomm or 0)
                self._record_trade(dt, 'SELL', order, pnl)
        
        self.order = None
    
    def _record_trade(self, dt, trade_type, order, pnl=0):
        self.trade_count += 1
        self.trade_records.append({
            'trade_id': self.trade_count,
            'datetime': dt,
            'type': trade_type,
            'price': order.executed.price,
            'size': order.executed.size,
            'value': order.executed.value,
            'commission': order.executed.comm,
            'portfolio_value': self.broker.getvalue(),
            'cash': self.broker.getcash(),
            'pnl': pnl,
        })
    
    def get_trade_records(self):
        return self.trade_records

