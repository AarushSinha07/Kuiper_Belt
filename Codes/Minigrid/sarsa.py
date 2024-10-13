import gym
import gym_minigrid
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("MiniGrid-Empty-6x6-v0")

alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
min_epsilon = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate of epsilon
tot_epi = 1000  # Number of episodes
max_steps = 100  # Max steps per episode

# Initialize Q-value table arbitrarily (e.g., to zeros)
Q = {}
for x in range(env.grid.width):
    for y in range(env.grid.height):
        for dir in range(4):
            s = (x, y, dir)
            Q[s] = np.zeros(env.action_space.n)

# Epsilon-greedy policy
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

# Training loop
epi_returns = []  # To store total returns per episode
epi_length=[]

for episode in range(tot_epi):
    obs, _ = env.reset()
    state = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]),obs["direction"])
    action = epsilon_greedy_policy(state, epsilon)
    total_reward = 0
    count =0

    for step in range(max_steps):
        obs, reward, done, _, _ = env.step(action)
        next_state = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]),obs["direction"])
        next_action = epsilon_greedy_policy(next_state, epsilon)

        # SARSA Update
        Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

        state, action = next_state, next_action
        total_reward += reward
        count+=1

        if done:
            break

    # Decay epsilon after each episode
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epi_returns.append(total_reward)
    epi_length.append(count)

plt.figure(figsize=(12, 5))
plt.suptitle("SARSA")
plt.subplot(1, 2, 1)
plt.plot(epi_length)
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.subplot(1, 2, 2)
plt.plot(epi_returns)
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.title('Returns per Episode')
plt.tight_layout()
plt.show()

