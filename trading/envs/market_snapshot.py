import datetime
from enum import Enum, auto

class MarketSnapshot():
    epochseconds = 0
    market = ''
    symbol = ''
    step_change = 0.0
    price_at_analysis = 0.0
    min_drop = 0.0
    max_jump = 0.0

    def __str__(self):
        return 'epochseconds: {epochseconds}, market: {market}, symbol: {symbol}, step_change: {step_change}, price_at_analysis: {price_at_analysis}, min_drop: {min_drop}, max_jump: {max_jump}'.format(
            epochseconds=self.epochseconds, market=self.market, symbol=self.symbol, step_change=self.step_change, price_at_analysis=self.price_at_analysis, min_drop=self.min_drop, max_jump=self.max_jump)

class TradeSideType(Enum):
    LONG = auto()
    SHORT = auto()

class TradeSnapshot():
    market_snapshot_enter = None
    market_snapshot_exit = None
    trade_side_type = None
    slippage = 0.0
    commission = 0.0

    def get_position_duration(self):
        seconds = self.market_snapshot_exit.epochseconds - self.market_snapshot_enter.epochseconds
        return datetime.timedelta(seconds=seconds)

    def get_profit(self):
        p = (self.market_snapshot_exit.price_at_analysis - self.market_snapshot_enter.price_at_analysis) / self.market_snapshot_enter.price_at_analysis
        if self.trade_side_type == TradeSideType.LONG:
            return p
        else:
            return -p

    def __str__(self):
        return 'side: {s}, profit: {profit}, duration: {du}'.format(s=self.trade_side_type, profit=self.get_profit(), du=self.get_position_duration())

class TradeSnapshots():
    trade_snapshots = []
    market_snapshot_enter = None
    current_trade_side_type = None
    slippage = 0.0
    commission = 0.0

    def open_trade(self, market_snapshot_enter: MarketSnapshot, trade_side_type: TradeSideType):
        self.market_snapshot_enter = market_snapshot_enter
        self.current_trade_side_type = trade_side_type

    def close_trade(self, market_snapshot_exit: MarketSnapshot):
        trade_snapshot = TradeSnapshot()
        trade_snapshot.market_snapshot_enter = self.market_snapshot_enter
        trade_snapshot.market_snapshot_exit = market_snapshot_exit
        trade_snapshot.trade_side_type = self.current_trade_side_type
        trade_snapshot.slippage = self.slippage
        trade_snapshot.commission = self.commission
        self.trade_snapshots.append(trade_snapshot)

    def print_summary(self):
        print('# of trades: {n}'.format(n=len(self.trade_snapshots)))
        print('total (raw) profit: {p}'.format(p=sum(map(lambda shot: shot.get_profit(), self.trade_snapshots))))

    def reset(self):
        self.trade_snapshots.clear()