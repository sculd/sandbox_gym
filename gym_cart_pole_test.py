import gymnasium as gym

# Initialise the environment
env = gym.make("CartPole-v1", render_mode="human")

import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="cart_pole",
)

import numpy as np
from collections import deque
import os
import agent

# Get state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print('State size: {}, action size: {}'.format(state_size, action_size))
dqn_agent = agent.DQNAgent(state_size, action_size, seed=0)

checkpoint_filename = f'cart_pole_solved_{agent.ENV_SOLVED}.pth'
if os.path.exists(checkpoint_filename):
    dqn_agent.load(checkpoint_filename)
else:
    env.close()
    exit(0)

scores_window = deque(maxlen=100)
max_test_episodes = 100
for episode in range(1, max_test_episodes + 1):
    state = env.reset()[0]
    score = 0
    for t in range(agent.MAX_STEPS):
        action = dqn_agent.act(state, eps=0.0)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        score += reward
        if done:
            break

    scores_window.append(score)
    mean_score = np.mean(scores_window)
    print('\r Progress {}/{}, average score:{:.2f}'.format(episode, max_test_episodes, mean_score), end="")

env.close()


