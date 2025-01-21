import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

import numpy as np
from collections import deque
import os
import agent

# Get state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print('State size: {}, action size: {}'.format(state_size, action_size))
dqn_agent = agent.DQNAgent(state_size, action_size, seed=0)

checkpoint_filename = f'solved_{agent.ENV_SOLVED}.pth'
if os.path.exists(checkpoint_filename):
    dqn_agent.load(checkpoint_filename, eval=True)
else:
    env.close()
    exit(0)

scores_window = deque(maxlen=100)
for episode in range(1, 100 + 1):
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
    print('\r Progress {}/{}, average score:{:.2f}'.format(episode, agent.MAX_EPISODES, mean_score), end="")

env.close()


