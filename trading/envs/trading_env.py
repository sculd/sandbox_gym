import numpy as np
import gym
from gym import spaces
from enum import Enum, auto

from trading.envs.train_test_data import MarketData, TrainTestDataType
from trading.envs.market_snapshot import MarketSnapshot, TradeSnapshots, TradeSnapshot, TradeSideType

DEFAULT_TEST_SPLIT = 0.4
DEFAULT_INITIAL_BALANCE = 1000
DEFAULT_BET_AMPLITUDE = 0.1
DEFAULT_TRADING_PRICE_SLIPPAGE = 0.002
DEFAULT_TRADING_COMMISION = 0.004
# epochSeconds,market,symbol,stepChange,priceAtAnalysis,minDrop,maxJump,changeSinceMinDrop,changeSinceMaxJump,normalizedPricePT3H,changePT6H,rsiPT3H,bandwidthPT2H2,bandPercentPT2H2,moneyFlowPT40M
DEFAULT_FEATURES = ['minDrop', 'maxJump', 'bandwidthPT2H2', 'bandPercentPT2H2', 'moneyFlowPT40M']
DEFAULT_SEED = 100

class StrategyTradeSideType(Enum):
    LONG = auto()
    SHORT = auto()
    LONG_SHORT = auto()

class TradingEnvParam():
    filename = 'data.csv'
    trade_side_type = StrategyTradeSideType.LONG
    test_split = DEFAULT_TEST_SPLIT
    initial_balance = DEFAULT_INITIAL_BALANCE
    bet_amplitude = DEFAULT_BET_AMPLITUDE
    trading_price_slippage = DEFAULT_TRADING_PRICE_SLIPPAGE
    trading_commission = DEFAULT_TRADING_COMMISION
    features = DEFAULT_FEATURES
    seed = DEFAULT_SEED

    def __str__(self):
        return 'filename: {filename}, side type: {st}, train test split: {sp}'.format(filename=self.filename, st=self.trade_side_type, sp=self.test_split)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    env_param:TradingEnvParam = None
    action_value_neutral_position: int = 0
    train_data: MarketData = None
    num_position_change: int = 0
    trade_snapshots_epoch: TradeSnapshots = None
    trade_snapshots: TradeSnapshots = None

    def __init__(self, env_param: TradingEnvParam):
        super(TradingEnv, self).__init__()

        self.env_param = env_param
        print(env_param)
        if env_param.trade_side_type == StrategyTradeSideType.LONG:
            self.action_value_neutral_position = 0
            space_cardinality = 2
        elif env_param.trade_side_type == StrategyTradeSideType.SHORT:
            self.action_value_neutral_position = 1
            space_cardinality = 2
        else:
            self.action_value_neutral_position = 1
            space_cardinality = 3

        self.train_data = MarketData(filename=env_param.filename, test_split=env_param.test_split, seed=env_param.seed)
        # 0, 1, 2 translates to 1, 0, s-1 for long, neutral, short
        self.action_space = spaces.Discrete(space_cardinality)
        # position + features
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1 + len(self.env_param.features),), dtype=np.uint8)

        self.num_position_change = 0
        self.trade_snapshots_epoch = TradeSnapshots()
        self.trade_snapshots_epoch.slippage = self.env_param.trading_price_slippage
        self.trade_snapshots_epoch.commission = self.env_param.trading_commission

        self.trade_snapshots = TradeSnapshots()
        self.trade_snapshots.slippage = self.env_param.trading_price_slippage
        self.trade_snapshots.commission = self.env_param.trading_commission

        self.reset()

    def _reward(self, action: int):
        step_change = self.entry['stepChange']
        profit = step_change * (action - self.action_value_neutral_position)
        # slippage causes the position to be positioned with a penalty
        if profit >= 0:
            profit *= 1.00 - self.env_param.trading_price_slippage
        else:
            profit *= 1.00 + self.env_param.trading_price_slippage
        action_amplitude = abs(action - self.prev_action)
        r = profit - action_amplitude * self.env_param.trading_commission
        return r * 100

    def _observation(self, action):
        # features
        obs = [action - self.action_value_neutral_position]
        for feature in self.env_param.features:
            obs.append(self.entry[feature])
        return np.array(obs)

    def _info(self):
        return {
            'balance': self.balance
        }

    def _next_entry(self):
        try:
            row = next(self.train_data)
            entry = {}
            for column, idx in self.train_data.column_name_to_idx.items():
                if column == 'epochSeconds':
                    entry[column] = int(row[idx])
                elif column == 'market' or column == 'symbol':
                    entry[column] = row[idx]
                else:
                    entry[column] = float(row[idx])
            return entry
        except StopIteration:
            return None

    def set_train_test(self, train_test_data_type, if_reset=True):
        self.train_data.set_train_test(train_test_data_type, if_reset=if_reset)

    def _entry_to_market_snapshot(self):
        market_snapshot = MarketSnapshot()
        market_snapshot.epochseconds, market_snapshot.market, market_snapshot.symbol, market_snapshot.step_change, market_snapshot.price_at_analysis = \
            self.entry['epochSeconds'], self.entry['market'], self.entry['symbol'], self.entry['stepChange'], self.entry['priceAtAnalysis']
        market_snapshot.min_drop, market_snapshot.max_jump = self.entry['minDrop'], self.entry['maxJump']
        for feature in self.env_param.features:
            market_snapshot.features[feature] = self.entry[feature]
        return market_snapshot

    def _record_action(self, action: int):
        if action == self.prev_action:
            return
        self.num_position_change += 1
        market_snapshot = self._entry_to_market_snapshot()
        # enter short
        if action == self.action_value_neutral_position - 1:
            self.trade_snapshots_epoch.open_trade(market_snapshot, TradeSideType.SHORT)
        # enter long
        elif action == self.action_value_neutral_position + 1:
            self.trade_snapshots_epoch.open_trade(market_snapshot, TradeSideType.LONG)
        # exit
        else:
            self.trade_snapshots_epoch.close_trade(market_snapshot)
            if self.train_data.train_test_data_type == TrainTestDataType.TEST:
                #print('num_position_change: {n}'.format(n=self.num_position_change))
                #print(self.trade_snapshots.trade_snapshots[-1])
                pass

    def step(self, action: int):
        # new market/symbol, reset the position
        market, symbol = self.entry['market'], self.entry['symbol']
        if market + symbol != self.prev_market_symbol:
            self.prev_action = self.action_value_neutral_position
        self.prev_market_symbol = market + symbol

        self._record_action(action)
        self.prev_action = action

        # action 0 to 2 translates to -1 to 1
        step_reward = self._reward(action)
        self.balance += self.balance * self.env_param.bet_amplitude * step_reward
        reward = self.balance / self.env_param.initial_balance - 1.0

        obs = self._observation(action)

        # read a row ahead to calculate reward in the next step
        self.entry = self._next_entry()

        done = self.entry is None
        if done and self.train_data.train_test_data_type == TrainTestDataType.TEST:
            self.trade_snapshots_epoch.print_summary()
            print('current epoch is done. num_position_change in the epoch: {v}'.format(v=self.num_position_change))
            print('')

        return obs, step_reward, done, self._info()

    def reset(self, **kwargs):
        self.trade_snapshots.merge(self.trade_snapshots_epoch)
        self.trade_snapshots_epoch.reset()
        self.num_position_change = 0
        self.prev_action = self.action_value_neutral_position
        self.prev_market_symbol = ''
        self.balance = self.env_param.initial_balance

        # the first row for the first observation
        self.entry = self._next_entry()
        obs = self._observation(self.action_value_neutral_position)
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
