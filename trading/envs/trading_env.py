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

        self.csvreader = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
        # header
        next(self.csvreader)

        # 1, 0, -1 for long, neutral, short
        self.action_space = spaces.Discrete(3)

        # minDrop,maxJump,changePT60H,rsiPT30M
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.uint8)

        self.prev_action = 0
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

    def step(self, action):
        try:
            entry = next(self.csvreader)
        except StopIteration:
            return None, 0, True, {}

        step_change = float(entry[3])
        reward = step_change * self.prev_action
        self.balance += self.balance * BET_AMPLITUDE * reward
        done = False

        return self._next_observation(entry), reward, done, self._info()

    def reset(self, **kwargs):
        self.prev_action = 0
        self.balance = INITIAL_BALANCE

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
