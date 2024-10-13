import gym
import math
import random
import gym_minigrid
import numpy as np
from matplotlib import pyplot as plt
env = gym.make("MiniGrid-Empty-6x6-v0")
epsilon = 0.8
gamma=.88
alpha=0.01
decay_rate=0.0001
tot_epi = 100
max_steps = 100 
j=0
val_function = {}
returns = {}
policy = {}
epi_length = []
epi_returns = []

for x in range(env.grid.width):
    for y in range(env.grid.height):
        for dir in range(4):
            s = (x, y, dir)
            val_function[s] = np.random.rand(env.action_space.n)  
            returns[s] = {a: [] for a in range(env.action_space.n)}  
            policy[s] = np.ones(env.action_space.n) *epsilon/ env.action_space.n        

def epsilon_policy(val_function, state, epsilon):
    if np.random.rand() < epsilon:
        return random.randint(0,2)
    else:
        return np.argmax(val_function[state]) 

def generate_episode(env, policy):
    episode = []
    current_state, _ = env.reset()
    dir = current_state["direction"]
    total_reward = 0
    no_of_steps = 0
    done = False

    while True and no_of_steps < max_steps:
        pos = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]), dir)
        action = policy(pos) 
        obs, reward, done, trunc, info = env.step(action)
        episode.append((pos, action, reward))
        total_reward += reward
        dir = obs["direction"]
        no_of_steps += 1
        current_state=obs
        if done:
            break
    return episode, total_reward

for num_epi in range(tot_epi):
    policy_func = lambda state: epsilon_policy(val_function, state, epsilon)
    epis, tot_reward = generate_episode(env, policy_func)
    epi_length.append(len(epis))
    epi_returns.append(tot_reward)
    sum = 0  
    for state, action, reward in reversed(epis):  
        sum = reward + (sum*gamma)
        returns[state][action].append(sum) 
        val_function[state][action]+= (sum - val_function[state][action])*alpha 

    for state in val_function.keys():
        probable_action = np.argmax(val_function[state])
        for action in range(env.action_space.n):
            if action == probable_action:
                policy[state][action] = 1 - epsilon + epsilon / env.action_space.n
                epsilon=epsilon*math.exp(-decay_rate*num_epi)
            else:
                policy[state][action] = epsilon / env.action_space.n
                epsilon=epsilon*math.exp(-decay_rate*num_epi)

plt.figure(figsize=(12, 5))
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
