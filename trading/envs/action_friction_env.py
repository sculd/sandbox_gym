import gym
import gym
import numpy as np
from gym import spaces

INITIAL_BALANCE = 1000
BET_AMPLITUDE = 0.1
TRADING_FRICTION = 0.004
ACTION_VALUE_NEUTRAL_POSITION = 1

class ActionFrictionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_entries):
        super(ActionFrictionEnv, self).__init__()

        self.num_entries = num_entries

        # 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.uint8)

        self.i = 0
        self.num_position_change = 0
        self.prev_action = ACTION_VALUE_NEUTRAL_POSITION
        self.balance = INITIAL_BALANCE
        self.reset()

    def _reward(self, action):
        action_amplitude = abs(action - self.prev_action)
        return - action_amplitude * TRADING_FRICTION

    def _observation(self, action):
        return np.array([
            action - 1
        ])

    def _info(self):
        return {
            'balance': self.balance
        }

    def step(self, action):
        # action 0 to 2 translates to -1 to 1
        reward = self._reward(action)
        if action != self.prev_action:
            self.num_position_change += 1
            pass
        self.prev_action = action
        self.balance += self.balance * BET_AMPLITUDE * reward

        obs = self._observation(action)
        self.i += 1
        return obs, reward, self.i >= self.num_entries, self._info()

    def reset(self, **kwargs):
        print('num_position_change: {v}'.format(v=self.num_position_change))
        self.i = 0
        self.num_position_change = 0
        self.prev_action = ACTION_VALUE_NEUTRAL_POSITION
        self.balance = INITIAL_BALANCE

        return self._observation(ACTION_VALUE_NEUTRAL_POSITION)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
