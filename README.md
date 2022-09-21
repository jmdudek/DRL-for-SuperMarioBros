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

---
# Model Architecture

<p float="left">
  <img src="https://github.com/jmdudek/DRL-for-SuperMarioBros/blob/main/Visualizations/model_architecture.jpg" width="300" />
</p>

--- 
# Results

---
# Video Presentation

[Click here](https://myshare.uni-osnabrueck.de/f/58616b38b8584804a4bc/) to get to a video presentation of our project held by Thorsten Krause.
