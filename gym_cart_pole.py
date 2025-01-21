import gymnasium as gym

# Initialise the environment
env = gym.make("CartPole-v1", render_mode=None)

import wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="cart_pole",
)

import numpy as np
from collections import deque
import time, sys, os
import agent

# Get state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print('State size: {}, action size: {}'.format(state_size, action_size))
dqn_agent = agent.DQNAgent(state_size, action_size, seed=0)

checkpoint_filename = f'cart_pole_solved_{agent.ENV_SOLVED}.pth'
if os.path.exists(checkpoint_filename):
    dqn_agent.load(checkpoint_filename)

start = time.time()
scores = []
# Maintain a list of last 100 scores
scores_window = deque(maxlen=100)
eps = agent.EPS_START
for episode in range(1, agent.MAX_EPISODES + 1):
    state = env.reset()[0]
    score = 0
    for t in range(agent.MAX_STEPS):
        action = dqn_agent.act(state, eps)
        next_state, reward, done, truncated, info = env.step(action)
        dqn_agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

        eps = max(eps * agent.EPS_DECAY, agent.EPS_MIN)
        if episode % agent.PRINT_EVERY == 0:
            mean_score = np.mean(scores_window)
            print('\r Progress {}/{}, average score:{:.2f}'.format(episode, agent.MAX_EPISODES, mean_score), end="")
        if score >= agent.ENV_SOLVED:
            mean_score = np.mean(scores_window)
            print('\rEnvironment solved in {} episodes, average score: {:.2f}'.format(episode, mean_score), end="")
            sys.stdout.flush()
            dqn_agent.checkpoint(checkpoint_filename)
            break

    wandb.log({"score": score, "eps": eps})
    scores_window.append(score)
    scores.append(score)

end = time.time()    
print('Took {} seconds'.format(end - start))

dqn_agent.checkpoint(checkpoint_filename)

env.close()


