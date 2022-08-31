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
