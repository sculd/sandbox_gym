import datetime
from enum import Enum, auto
from typing import List

class MarketSnapshot():
    epochseconds = 0
    market = ''
    symbol = ''
    step_change = 0.0
    price_at_analysis = 0.0
    min_drop = 0.0
    max_jump = 0.0
    features = {}

    def __init__(self):
        self.features = {}

    def to_csv_header(self, prefix = None) -> str:
        if not prefix:
            prefix = ''
        l = [prefix + 'epochseconds', prefix + 'market', prefix + 'symbol']
        l += [prefix + k for k in self.features.keys()]
        return ','.join(l)

    def to_csv_row(self) -> str:
        l = ['{}'.format(self.epochseconds), self.market, self.symbol]
        l += [str(v) for v in self.features.values()]
        return ','.join(l)


    def __str__(self):
        return 'epochseconds: {epochseconds}, market: {market}, symbol: {symbol}, step_change: {step_change}, price_at_analysis: {price_at_analysis}, min_drop: {min_drop}, max_jump: {max_jump}, features: {features}'.format(
            epochseconds=self.epochseconds, market=self.market, symbol=self.symbol, step_change=self.step_change, price_at_analysis=self.price_at_analysis, min_drop=self.min_drop, max_jump=self.max_jump, features=self.features)

class TradeSideType(Enum):
    LONG = auto()
    SHORT = auto()

class TradeSnapshot():
    market_snapshot_enter: MarketSnapshot = None
    market_snapshot_exit: MarketSnapshot = None
    trade_side_type: TradeSideType = None
    slippage: float = 0.0
    commission: float = 0.0

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

    def to_csv_header(self) -> str:
        return '{enter},{exit},trade_side_type,profit,duration'.format(enter=self.market_snapshot_enter.to_csv_header('enter.'), exit=self.market_snapshot_exit.to_csv_header('exit.'))

    def to_csv_row(self) -> str:
        return '{enter},{exit},{side},{profit},{dur}'.format(
            enter=self.market_snapshot_enter.to_csv_row(), exit=self.market_snapshot_exit.to_csv_row(),
            side=self.trade_side_type, profit=self.get_profit(), dur=self.get_position_duration())

    def __str__(self):
        return 'symbol: {symbol}, profit: {profit}, duration: {du} side: {s},\nenter: {enter}\nexit: {exit}'.format(
            enter=self.market_snapshot_enter, exit=self.market_snapshot_exit, symbol=self.market_snapshot_enter.symbol, s=self.trade_side_type, profit=round(self.get_profit(), 3), du=self.get_position_duration())

class TradeSnapshots():
    trade_snapshots: List[TradeSnapshot] = None
    open_trade_snapshot: TradeSnapshot = None
    slippage: float = 0.0
    commission: float = 0.0

    def __init__(self):
        self.trade_snapshots = []

    def open_trade(self, market_snapshot_enter: MarketSnapshot, trade_side_type: TradeSideType):
        self.market_snapshot_enter = market_snapshot_enter

        self.open_trade_snapshot = TradeSnapshot()
        self.open_trade_snapshot.market_snapshot_enter = market_snapshot_enter
        self.open_trade_snapshot.trade_side_type = trade_side_type
        self.open_trade_snapshot.slippage = self.slippage
        self.open_trade_snapshot.commission = self.commission

    def close_trade(self, market_snapshot_exit: MarketSnapshot):
        trade_snapshot = TradeSnapshot()
        trade_snapshot.market_snapshot_enter = self.open_trade_snapshot.market_snapshot_enter
        trade_snapshot.market_snapshot_exit = market_snapshot_exit
        trade_snapshot.trade_side_type = self.open_trade_snapshot.trade_side_type
        trade_snapshot.slippage = self.slippage
        trade_snapshot.commission = self.commission
        self.trade_snapshots.append(trade_snapshot)

    def print_summary(self):
        print('epoch summary')
        print('# of trades: {n}'.format(n=len(self.trade_snapshots)))
        if self.trade_snapshots:
            print('total profit: {p}, max: {mp}, min: {mnp}'.format(
                p=round(sum(map(lambda shot: shot.get_profit(), self.trade_snapshots)), 4),
                mp=round(max(map(lambda shot: shot.get_profit(), self.trade_snapshots)), 4),
                mnp=round(min(map(lambda shot: shot.get_profit(), self.trade_snapshots)), 4)
            ))

            print('duration, avg: {avg}, max: {md}, min: {mnd}'.format(
                avg=datetime.timedelta(seconds=sum(map(lambda shot: shot.get_position_duration().total_seconds(), self.trade_snapshots))/len(self.trade_snapshots)),
                md=datetime.timedelta(seconds=max(map(lambda shot: shot.get_position_duration().total_seconds(), self.trade_snapshots))),
                mnd=datetime.timedelta(seconds=min(map(lambda shot: shot.get_position_duration().total_seconds(), self.trade_snapshots)))
            ))

            print('last trade:\n{t}'.format(t=self.trade_snapshots[-1]))
            if len(self.trade_snapshots) > 1:
                print('second last trade:\n{t}'.format(t=self.trade_snapshots[-2]))
            if len(self.trade_snapshots) > 2:
                print('third last trade:\n{t}'.format(t=self.trade_snapshots[-3]))
        else:
            print('no closed trade made')

        if self.market_snapshot_enter:
            print('open position enter: {t}'.format(t=self.market_snapshot_enter))

    def to_csv(self, filename):
        if not self.trade_snapshots:
            return
        with open(filename, 'w') as f:
            f.write('{}\n'.format(self.trade_snapshots[0].to_csv_header()))
            for trade_snapshot in self.trade_snapshots:
                f.write('{}\n'.format(trade_snapshot.to_csv_row()))

    def merge(self, other):
        assert self.slippage == other.slippage
        assert self.commission == other.commission
        self.trade_snapshots += other.trade_snapshots

    def reset(self):
        self.trade_snapshots.clear()
        self.open_trade_snapshot = None
