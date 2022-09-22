# About this Repository

This repository contains the code and paper to our (Gerrit Bartels, Thorsten Krause and Jacob Dudek) project in Deep Reinforcment Learning. We investigated benefits of Transfer Learning regarding Soft Q-Networks and Double Deep Q-Networks, as well as an alledged relationship between overfitting and negative transfer. We propose and executed an experimental setup, provide a ready-to-use implementation and identified and put forth major challenges that future research can build upon. 

Currently under construction. Please try reloading until the project is finished. 

---
# The Environment

As test bed we used the popular NES game "Super Mario Bros.". The game consists of 32 levels in which the player has to control Super Mario through a parkour of obstacles by choosing from 256 distinct actions. We relied on a ready-to-use implementation that can be found [here](https://pypi.org/project/gym-super-mario-bros/).

For transfer learning we chose level 1-1 (left) as the source and level 1-2 (right) as the target domain. Below are exemplary scenes from both levels.

<p float="left">
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/Visualizations/level_1_1.png" width="300" />
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/Visualizations/level_1_2.png" width="300" />
</p>

## Preprocessing
The agents received the game state as a normalized, rescaled 84x84 grey-scale picture and drew from a restricted action space of five actions: (1) *idle*, (2) *move right*, (3) *jump right*, (4) *move right and throw a fire ball*, (5) *jump right and throw a fireball*. 
As consecutive frames are highly correlated, we accelerated training by repeating each action over four frames and passing the corresponding states as a stacked 4x84x84 image.

---
# Model Architecture

The following figure visualizes our CNN backbone architecture employed in both our DDQN and SoftQN.

<p float="left">
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/Visualizations/model_architecture.jpg" width="300" />
</p>

---
# Performance on Level 1-1

DDQN:
<p float="left">
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/DDQN%20Results/DDQN_1-1_rewards.png" height="220" />
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/DDQN%20Results/DDQN_1-1_wins.png" height="220" />
</p>

SoftQN:
<p float="left">
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/SoftQN%20Results/SoftQ_1-1_rewards.png" height="220" >
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/SoftQN%20Results/SoftQ_1-1_wins.png" height="220" />
</p>

--- 
# Performance on Level 1-2

DDQN:
<p float="left">
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/DDQN%20Results/DDQN_1-2_rewards.png" height="220" />
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/DDQN%20Results/DDQN_1-2_wins.png" height="220" />
</p>

SoftQN:
<p float="left">
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/SoftQN%20Results/SoftQ_1-2_rewards.png" height="220" >
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/SoftQN%20Results/SoftQ_1-2_wins.png" height="220" />
</p>

---
# Video Presentation

[Click here](https://myshare.uni-osnabrueck.de/f/58616b38b8584804a4bc/) to get to a video presentation of our project held by Thorsten Krause.
