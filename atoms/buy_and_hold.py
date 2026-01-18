"""
买入并持有基准策略
"""
import backtrader as bt
from bt_base import StrategyAtom, BaseStrategy


class BuyAndHoldAtom(StrategyAtom):
    """
    买入并持有基准策略

    在第一个bar用所有资金买入并持有到最后
    适合作为基准策略，评估主动交易策略的表现
    """

    name = "buy_and_hold"
    params = {}

    def strategy_cls(self):
        class BuyAndHoldStrategy(BaseStrategy):
            """
            买入并持有策略实现

            第一个bar用所有可用资金买入，最后一个bar卖出
            这样可以完整记录投资周期的成本和收益
            """
            def __init__(self):
                super().__init__()
                self.order = None
                self.bought = False

            def next(self):
                # 只在第一次且无挂单时买入
                if not self.bought and not self.order:
                    # 使用所有可用资金
                    cash = self.broker.getcash()
                    price = self.data.close[0]
                    # 计算能买多少手（向下取整）
                    size = int(cash / price)

                    if size > 0:
                        self.order = self.buy(size=size)
                        self.bought = True
                        self.log(f'买入指令已发出: 价格={price:.2f}, 数量={size}')

            def notify_order(self, order):
                """订单状态通知"""
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.log(f'买入完成: 价格={order.executed.price:.2f}, 数量={order.executed.size}, 手续费={order.executed.comm:.2f}')
                    elif order.issell():
                        self.log(f'卖出完成: 价格={order.executed.price:.2f}, 数量={order.executed.size}, 手续费={order.executed.comm:.2f}')
                    self.order = None

                elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                    self.log(f'订单异常: {order.status}')
                    self.order = None

            def stop(self):
                """回测结束时卖出仓位"""
                if self.position and self.bought and not self.order:
                    # 在最后一个bar卖出
                    self.order = self.sell(size=self.position.size)
                    self.log(f'终期卖出指令已发出: 价格={self.data.close[0]:.2f}, 数量={self.position.size}')

        return BuyAndHoldStrategy

    def sizer_cls(self):
        """不使用额外的Sizer，在策略内部计算仓位"""
        return None
