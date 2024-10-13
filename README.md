# Kuiper Belt Using Reinforcement Learning


## Aim
The objective of the project was to stimulate and analyse an agent's behaviour in the Kuiper belt environment and hence train it to maximise its reward in the environment using reinforcement learning.


## Libraries
1. Numpy
2. Matplotlib
3. Gymnasium
   - gym_minigrid
   - gym_kuiper_escape

#### Frozen lake environment
To grasp the basics, it was necessary to implement initial problems and algorithms of reinforcement learning on a simpler environment, and the Frozen Lake environment was the perfect choice. In this environment, there was complete knowledge of its dynamics, allowing the algorithms to be designed with these specifics in mind, resulting in the best possible solution.

![Frozen Lake Env](Images/Frozen_Lake.png)

###### State Space
Frozen lake involves crossing a frozen lake from Start to Goal without falling into any Holes by walking over the Frozen lake. The agent may not always move in the intended direction due to the slippery nature of the frozen lake.The observation is a value representing the agent’s current position as current_row * nrows + current_col.

###### Action Space
The agent takes a 1-element vector for actions. The action space is (dir), where dir decides direction to move in which can be:
   - 0: LEFT
   - 1: DOWN
   - 2: RIGHT
   - 3: UP

###### Reward Function
Reward schedule:
   - Reach goal: +1
   - Reach hole: 0
   - Reach frozen: 0

###### Algorithms
In the Frozen Lake environment, Dynamic Programming (DP) techniques are employed to derive an optimal policy for navigating through the grid. Specifically, the algorithms used include Policy Iteration and Value Iteration, both of which leverage the principles of dynamic programming to solve Markov Decision Processes (MDPs).

The link for the code has been provided below .



Here's a GIF demonstrating the Frozen Lake algorithm:
![Frozen Lake](Gifs/Frozen_Lake_gif.gif)
