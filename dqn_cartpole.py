import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'CartPole-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

import wandb
from wandb.keras import WandbCallback

wandb.init(project="cart-pole", entity="trading-rl")
memory_limit = 1000
dqn_target_model_update=1e-2
adam_lr=1e-3
adam_metrics='mae'
dqn_nb_steps=50000

wandb.config = {
  "memory_limit": memory_limit,
  "dqn_target_model_update": dqn_target_model_update,
  "adam_lr": adam_lr,
  "adam_metrics": adam_metrics,
  "dqn_nb_steps": dqn_nb_steps,
}

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=memory_limit, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=dqn_target_model_update, policy=policy)
dqn.compile(Adam(lr=adam_lr), metrics=[adam_metrics])

dqn.fit(env, nb_steps=dqn_nb_steps, visualize=True, verbose=2, callbacks=[WandbCallback()])

print('test for 5 episodes')
dqn.test(env, nb_episodes=5, visualize=True)

