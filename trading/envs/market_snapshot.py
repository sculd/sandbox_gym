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
        step_change = (self.market_snapshot_exit.price_at_analysis - self.market_snapshot_enter.price_at_analysis) / self.market_snapshot_enter.price_at_analysis
        if self.trade_side_type == TradeSideType.LONG:
            action = 1
        else:
            action = -1

        profit = step_change * action
        # slippage causes the position to be positioned with a penalty
        if profit >= 0:
            profit *= 1.00 - self.slippage
        else:
            profit *= 1.00 + self.slippage
        return profit - self.commission

    def __str__(self):
        return 'enter: {enter}, exit: {exit}, symbol: {symbol}, side: {s}, profit: {profit}, duration: {du}'.format(
            enter=self.market_snapshot_enter, exit=self.market_snapshot_exit, symbol=self.market_snapshot_enter.symbol, s=self.trade_side_type, profit=self.get_profit(), du=self.get_position_duration())

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
        print('total (raw) profit: {p}, max: {mp}, min: {mnp}'.format(
            p=round(sum(map(lambda shot: shot.get_profit(), self.trade_snapshots)), 4),
            mp=round(max(map(lambda shot: shot.get_profit(), self.trade_snapshots)), 4),
            mnp=round(min(map(lambda shot: shot.get_profit(), self.trade_snapshots)), 4)
        ))

    def reset(self):
        self.trade_snapshots.clear()
        self.market_snapshot_enter = None
        self.current_trade_side_type = None
