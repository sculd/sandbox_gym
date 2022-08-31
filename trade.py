import gym
import random
import numpy as np
from trading.envs.trading_env import TradingEnv

filename = 'market_data_binance.by_minute_DGBUSDT-mldata.csv'
env = TradingEnv(filename)


states = env.observation_space.shape[0]
print('States', states)
actions = env.action_space.n
print('Actions', actions)

episodes = 10
for episode in range(1,episodes+1):
    # At each begining reset the game
    state = env.reset()
    # set done to False
    done = False
    # set score to 0
    score = 0
    # while the game is not finished
    while not done:
        # visualize each step
        env.render()
        # choose a random action
        action = random.choice([0,1])
        # execute the action
        n_state, reward, done, info = env.step(action)
        # keep track of rewards
        score+=reward
    print('episode {} score {}'.format(episode, score))


