import gym
import random
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, SimpleRNN
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from trading.envs.trading_env import TradingEnv, TradeSideType

filename = 'data_toy.csv'
env = TradingEnv(filename, TradeSideType.LONG)


print('Observatoin shape', env.observation_space.shape)
#'''
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#model.add(SimpleRNN(4, input_shape=env.observation_space.shape))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))
print(model.summary())
input("Press Enter to continue...")

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# You can always safely abort the training prematurely using Ctrl + C.
dqn.fit(env, nb_steps=90000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format('trading'), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
#'''
