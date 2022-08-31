# IDLLA: Improved Deep Localized Level Analysis
IDLLA is an approach for predicting player emotional responses to gameplay. 
The approach uses a convolutional neural network (CNN) to analyze sequences of gameplay event log information 
and associated level-structure information in order to predict how a player would rate that gameplay sequence.

The CNN learns to identify patterns in game information, and in combination with player demographic information, tries to predict how
such a player would rate the level being played in the sequence being analyzed. Ratings range from "most x," "mid x," and "least x," where
x represents any one of the following four metrics measured: "Fun, Frustration, Challenge, and Design."

This repository contains the codebase that was used for testing the IDLLA approach, and can be used to recreate our work.
Details on our approach used can be found in "Improving Deep Localized Level Analysis: How Game Logs Can Help," an upcoming workshop paper at EXAG 2022.
For citing this work, please use:

> Citation information will be uploaded once this work is published :P. Feel free to link this repo in the meantime.

This work utilized datasets from three other sources, which are cited here:
"Game Level Generation from Gameplay Videos" by Matthew Guzdial and Mark O Riedl from the Twelfth Artificial Intelligence and Interactive Digital Entertainment Conference. 
> http://guzdial.com/datasets/MarioPCGStudy.zip

"Evaluating Singleplayer and Multiplayer in Human Computation Games" by Kristin Siu, Matthew Guzdial, and Mark O Riedl from the Foundations of Digitial Games Conference 2017. 
> http://guzdial.com/datasets/GwarioData.zip

"The vglc: The video game level corpus." by Summerville, Adam James, Sam Snodgrass, Michael Mateas, and Santiago Ontanón. arXiv preprint arXiv:1606.07487 (2016).
> https://github.com/TheVGLC/TheVGLC

This work was conducted as part of the University of Alberta GRAIL Lab.

# Software requirements
This code uses the following software:
- Python 3.10.4
- tflearn
- tensorflow
- scipy
- numpy

# How to use this repository
## 1. Clone the repo
> 

