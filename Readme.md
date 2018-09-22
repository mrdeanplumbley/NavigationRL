# Reinforcement learning to solve the navigation problem

This repo presents three solutions to the continuous control navigation problem using a unity environment

## Learning Environment

![Banana Environment](./images/banana.gif)

The environment is a continuous control navigation problem; the goal is to collect the maximum number of yellow bananas in the area during a set maximum number of moves. 
To make things more of a challenge the play area also contains some blue bananas which provide a point penalty. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

There are four possible actions associated with the environment.

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

In order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

The state space has 37 dimensions and contains information about the agents proximity to the objects around it, along with its velocity.

## Vanilla DQN

The most basic implementation in this repo is the vanilla DQN. 

