import gym
import gym_minigrid
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("MiniGrid-Empty-6x6-v0")

alpha = 0.1  
gamma = 0.99  
epsilon = 1.0  
min_epsilon = 0.0001 
epsilon_decay = 0.995  
lambda_param = 0.9  
tot_epi = 1000 
max_steps = 100  

epi_returns = []
epi_length=[]
Q = {}
E = {}
for x in range(env.grid.width):
    for y in range(env.grid.height):
        for dir in range(4):
            s = (x, y, dir)
            Q[s] = np.zeros(env.action_space.n)
            E[s] = np.zeros(env.action_space.n)

def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
  
for episode in range(tot_epi):
    obs, _ = env.reset()
    state = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]),obs["direction"])
    action = epsilon_greedy_policy(state, epsilon)
    total_reward = 0
    count=0

    for s in E.keys():
        E[s] = np.zeros(env.action_space.n)

    for step in range(max_steps):
        obs, reward, done, _, _ = env.step(action)
        next_state = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]),obs["direction"])
        next_action = epsilon_greedy_policy(next_state, epsilon)

        delta = reward + gamma * Q[next_state][next_action] - Q[state][action]

        E[state][action] += 1

        for s in Q.keys():
            Q[s] += alpha * delta * E[s]
            E[s] *= gamma * lambda_param  
        state, action = next_state, next_action
        total_reward += reward
        count +=1

        if done:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    epi_returns.append(total_reward)
    epi_length.append(count)

plt.figure(figsize=(12, 5))
plt.suptitle("SARSA_LAMBDA")
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
