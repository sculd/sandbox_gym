import gym
import numpy as np
from gym import spaces
import csv
from enum import Enum, auto

INITIAL_BALANCE = 1000
BET_AMPLITUDE = 0.1
TRADING_FRICTION = 0.004

class TradeSideType(Enum):
    LONG = auto()
    SHORT = auto()
    LONG_SHORT = auto()

class TradingEnvInitParam():
    filename = ''
    trade_side_type = TradeSideType.LONG

    def __str__(self):
        return 'filename: {filename}, side type: {st}'.format(filename=self.filename, st=self.trade_side_type)

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, init_param):
        super(TradingEnv, self).__init__()

        print(init_param)
        self.filename = init_param.filename
        if init_param.trade_side_type == TradeSideType.LONG:
            self.action_value_neutral_position = 0
            space_cardinality = 2
        elif init_param.trade_side_type == TradeSideType.SHORT:
            self.action_value_neutral_position = 1
            space_cardinality = 2
        else:
            self.action_value_neutral_position = 1
            space_cardinality = 3

        # 0, 1, 2 translates to 1, 0, s-1 for long, neutral, short
        self.action_space = spaces.Discrete(space_cardinality)
        # minDrop,maxJump,changePT60H,rsiPT30M
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.uint8)

        self.num_position_change = 0
        self.reset()

    def _reward(self, action):
        step_change = float(self.entry[3])
        action_amplitude = abs(action - self.prev_action)
        return step_change * (action - self.action_value_neutral_position) - action_amplitude * TRADING_FRICTION

    def _observation(self, action):
        min_drop, max_jump, change_6h, rsi_30m = \
            float(self.entry[4]), float(self.entry[5]), float(self.entry[7]), float(self.entry[8])
        return np.array([
            action - self.action_value_neutral_position,  min_drop, max_jump #, change_6h, rsi_30m
        ])

    def _info(self):
        return {
            'balance': self.balance
        }

    def _next_entry(self):
        try:
            entry = next(self.csvreader)
            if entry is not None and entry[0] == 'epochSeconds':
                entry = next(self.csvreader)
            return entry
        except StopIteration:
            return None

    def step(self, action):
        # new market/symbol, reset the position
        market, symbol = self.entry[1], self.entry[2]
        if market + symbol != self.prev_market_symbol:
            #print('market symbol change to {n} from {p}'.format(n=market+symbol, p=self.prev_market_symbol))
            self.prev_action = self.action_value_neutral_position
        self.prev_market_symbol = market + symbol

        if action != self.prev_action:
            self.num_position_change += 1
            pass
        self.prev_action = action

        # action 0 to 2 translates to -1 to 1
        reward = self._reward(action)
        self.balance += self.balance * BET_AMPLITUDE * reward

        obs = self._observation(action)

        # read a row ahead to calculate reward in the next step
        self.entry = self._next_entry()

        return obs, reward, self.entry is None, self._info()

    def reset(self, **kwargs):
        self.csvreader = csv.reader(open(self.filename, newline=''), delimiter=',', quotechar='|')
        if self.num_position_change > 0:
            print('num_position_change: {v}'.format(v=self.num_position_change))

        self.num_position_change = 0
        self.prev_action = self.action_value_neutral_position
        self.prev_market_symbol = ''
        self.balance = INITIAL_BALANCE

        # the first row for the first observation
        self.entry = self._next_entry()
        obs = self._observation(self.action_value_neutral_position)
        return obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
