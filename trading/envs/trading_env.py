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
        self.csvreader = csv.reader(open(self.filename, newline=''), delimiter=',', quotechar='|')
        # header
        next(self.csvreader)

        # 1, 0, -1 for long, neutral, short
        self.action_space = spaces.Discrete(3)

        # minDrop,maxJump,changePT60H,rsiPT30M
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.uint8)

        self.prev_action = 0
        # first row
        self.entry = next(self.csvreader)
        self.balance = INITIAL_BALANCE
        self.reset()

    def _reward(self, entry):
        step_change = float(entry[3])
        return step_change * self.prev_action

    def _next_observation(self, entry):
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
            return next(self.csvreader)
        except StopIteration:
            return None

    def step(self, action):
        # the reward for the current action is decided in the next step
        step_change = float(self.entry[3])
        reward = step_change * self.prev_action
        self.balance += self.balance * BET_AMPLITUDE * reward

        obs = self._next_observation(self.entry)
        self.entry = self._next_entry()
        return obs, reward, self.entry is None, self._info()

    def reset(self, **kwargs):
        self.csvreader = csv.reader(open(self.filename, newline=''), delimiter=',', quotechar='|')
        # header
        next(self.csvreader)
        # first row
        self.entry = next(self.csvreader)
        self.prev_action = 0
        self.balance = INITIAL_BALANCE

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
