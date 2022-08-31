import gym
import numpy as np
from gym import spaces
import csv

INITIAL_BALANCE = 1000
BET_AMPLITUDE = 0.1


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, filename):
        super(TradingEnv, self).__init__()

        self.filename = filename

        # 0, 1, 2 translates to 1, 0, s-1 for long, neutral, short
        self.action_space = spaces.Discrete(3)
        # minDrop,maxJump,changePT60H,rsiPT30M
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.uint8)

        self.entry = None
        self.prev_action = 0
        self.prev_market_symbol = ''
        self.balance = INITIAL_BALANCE
        self.reset()

    def _reward(self, entry):
        step_change = float(entry[3])
        return step_change * self.prev_action

    def _observation(self, entry):
        min_drop, max_jump, change_6h, rsi_30m = \
            float(entry[4]), float(entry[5]), float(entry[7]), float(entry[8])
        return np.array([
            min_drop, max_jump, change_6h, rsi_30m
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
            print('market symbol change to {n} from {p}'.format(n=market+symbol, p=self.prev_market_symbol))
            self.prev_action = 0
        self.prev_market_symbol = market + symbol

        # the reward for the current action is decided in the next step
        step_change = float(self.entry[3])
        # action 0 to 2 translates to -1 to 1
        reward = step_change * (self.prev_action - 1) 
        self.prev_action = action
        self.balance += self.balance * BET_AMPLITUDE * reward

        obs = self._observation(self.entry)
        self.entry = self._next_entry()
        return obs, reward, self.entry is None, self._info()

    def reset(self, **kwargs):
        self.csvreader = csv.reader(open(self.filename, newline=''), delimiter=',', quotechar='|')
        # header
        next(self.csvreader)
        # first row
        entry = next(self.csvreader)
        self.entry = next(self.csvreader)
        # reserve the second row
        self.prev_action = 0
        self.prev_market_symbol = ''
        self.balance = INITIAL_BALANCE

        return self._observation(entry)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
