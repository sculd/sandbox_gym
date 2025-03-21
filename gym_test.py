import gymnasium as gym

import numpy as np
from collections import deque
import os
import dqn.agent

def test(env_name, env_args={}, max_episodes=100, max_steps=dqn.agent.MAX_STEPS):
    # Initialise the environment
    env = gym.make(env_name, **env_args, render_mode="human")

    # Get state and action sizes
    state_size = env.observation_space.shape[0]

    print(f'{state_size=}, {env.action_space=}')
    dqn_agent = dqn.agent.DQNAgent(state_size, env.action_space, seed=0, for_training=False)

    checkpoint_filename = f'{env_name}_solved_{dqn.agent.ENV_SOLVED}.pth'
    if os.path.exists(checkpoint_filename):
        dqn_agent.load(checkpoint_filename)
    else:
        env.close()
        exit(0)

    scores_window = deque(maxlen=100)
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_steps):
            action = dqn_agent.act(state, eps=0.0)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        mean_score = np.mean(scores_window)
        print('\r Progress {}/{}, average score:{:.2f}'.format(episode, max_episodes, mean_score), end="")

    env.close()

test("LunarLander-v3", env_args={"enable_wind": True, "wind_power": 15.0, "turbulence_power": 1.5})
#test("CartPole-v1", max_steps=200)
#test("Acrobot-v1", max_steps=2000)



