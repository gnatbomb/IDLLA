from pickle import GLOBAL
import numpy as np
import csv
import os
import glob
import shutil
import random

random.seed(0)

# Extracts the level information for the original mario levels into a one-hot array.
# Also creates empty "dummy logs" associated with each level.

# Retrieve level data
levelfilenames = os.listdir("../OriginalMario/levelstxt/")
numlevels = len(levelfilenames)

# get the shortest level length. We crop levels to the length of the shortest level.
levellen = 1000
for i in range(numlevels):
    level = open('../OriginalMario/levelstxt/' + levelfilenames[i], 'r')
    if levellen > len(level.readline()):
        levellen = len(level.readline())

# Create output array for levels.
onehot = np.zeros((numlevels, 10, levellen, 17), dtype=np.int8)

# Reminder for the indices used for the symbols. 13-15 represidnt bullet bill guns.
'''0 = empty space
1 = HILL_TOP
2 = GROUND
3 = ROCK
4 = TUBE_TOP_LEFT
5 = TUBE_TOP_RIGHT
6 = TUBE_SIDE_LEFT
7 = TUBE_SIDE_RIGHT
8 = BLOCK_EMPTY
9 = BLOCK_COIN
10 = BLOCK_POWERUP
11 = ENEMY_GOOMBA
12 = GREEN_KOOPA
13 = (byte) (14 + 0 * 16)
14 = (byte) (14 + 1 * 16)
15 = (byte) (14 + 2 * 16)
16 = COIN'''

# Conversions from the VGLC level notation to the notation our model uses.
symboldict = {
    '-': 0,
    'X': 3,
    '<': 4,
    '>': 5,
    '[': 6,
    ']': 7,
    'S': 8,
    'Q': 9,
    '?': 10,
    'E': 11,
    'b': 13,
    'B': 15,
    'o': 16,
}

# For each level, convert the txt format into our one-hot array.
for i in range(numlevels):
    level = open('../OriginalMario/levelstxt/' + levelfilenames[i], 'r')
    # Remove top four rows.
    for i in range(4):
        level.readline()

    for y in range(10):
        strip = level.readline()
        for x in range(levellen):
            if strip[x] != '\n':
                # if a block is on the bottom row, record it as a HILL_TOP, otherwise, record it as a rock.
                if y == 9 and strip[x] == 'X':
                    onehot[i][y][x][1] = 1
                else:
                    onehot[i][y][x][symboldict[strip[x]]] = 1


# Save one-hot array
np.save("OriginalMario/levels", onehot)


# make "log" values:
logs = np.zeros((numlevels * levellen, 41), dtype=np.int16)

# If fillvalues = 1, set log values randomly. If fillvalues = 0, leave logs empty.
# Our research used empty logs.
fillvalues = 0

# Create dummy logs for each level.
for lvl in range(numlevels):
    # Used to calculate index in the output array.
    offset = lvl * levellen

    # Create an entry for each x-position in the level.
    for x in range(levellen):
        # Mario moves
        for val in range(3,9):
            if random.random() < 0.1:
                logs[offset + x][val] = fillvalues

        # Mario state
        logs[offset + x][random.randint(8,11)] = fillvalues

        # deaths
        for val in range(11, 16):
            if random.random() < 0.00001:
                logs[offset + x][val] = fillvalues

        # Interactions
        for val in range(16, 31):
            if random.random() < 0.01:
                logs[offset + x][val] = fillvalues

        # Scores
        logs[offset + x][31] = random.randint(0, 2)
        logs[offset + x][32] = random.randint(0, 2)
        logs[offset + x][33] = random.randint(0, 2)

        # Demographics
        for demographic in range(35, 39):
            logs[offset + x][demographic] = random.randint(1, 5) * fillvalues

        # Level
        logs[offset + x][39] = lvl

        # Mario's X
        logs[offset + x][40] = x

# Output dummy logs to file.
np.savetxt("OriginalMario/logs.csv", logs, delimiter=",", fmt='%d')