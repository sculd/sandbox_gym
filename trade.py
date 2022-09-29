import gym
import random
import numpy as np


from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, SimpleRNN
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from trading.envs.trading_env import TradingEnv, StrategyTradeSideType, TradingEnvParam, TrainTestDataType
from trading.envs.market_snapshot import TradeSnapshots

import wandb
from wandb.keras import WandbCallback

filename = 'data.csv'
init_param = TradingEnvParam()
init_param.filename = filename
init_param.trade_side_type = StrategyTradeSideType.LONG
env = TradingEnv(init_param)


print('Action space', env.action_space)
print('Observatoin shape', env.observation_space.shape)
#'''
# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
#model.add(SimpleRNN(4, input_shape=env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(env.action_space.n))
model.add(Activation('linear'))
print(model.summary())
#input("Press Enter to continue...")


memory_limit = 1000
dqn_target_model_update=1e-2
adam_lr=1e-3
adam_metrics='mae'
dqn_nb_steps=3000000

wandb.init(project="long multi assets", entity="trading-rl")
wandb.config = {
  "memory_limit": memory_limit,
  "dqn_target_model_update": dqn_target_model_update,
  "adam_lr": adam_lr,
  "adam_metrics": adam_metrics,
  "dqn_nb_steps": dqn_nb_steps,
}

# ... Define a model
memory = SequentialMemory(limit=memory_limit, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=120, target_model_update=dqn_target_model_update, policy=policy)
dqn.compile(Adam(lr=adam_lr), metrics=[adam_metrics])

if False:
  # The training can be aborted prematurely using Ctrl + C.
  dqn.fit(env, nb_steps=dqn_nb_steps, visualize=False, verbose=2, callbacks=[WandbCallback()])

  dqn.save_weights('dqn_trading_weights.h5f', overwrite=True)
else:
  dqn.load_weights('dqn_trading_weights.h5f')


'''
print("Testing with training dataset")
dqn.test(env, nb_episodes=200, visualize=True)

print("Testing with test dataset")
env.set_train_test(TrainTestDataType.TEST)
dqn.test(env, nb_episodes=200, visualize=True)
env.reset()
'''

filename_validation = 'data_2.csv'
init_param_validation = TradingEnvParam()
init_param_validation.filename = filename_validation
init_param_validation.trade_side_type = StrategyTradeSideType.LONG
init_param_validation.test_split = 0.9
env_validation = TradingEnv(init_param_validation)
print("Testing with validation dataset")
env_validation.set_train_test(TrainTestDataType.TEST)
dqn.test(env_validation, nb_episodes=200, visualize=True)
env_validation.reset()


tradesnapshot = TradeSnapshots()
