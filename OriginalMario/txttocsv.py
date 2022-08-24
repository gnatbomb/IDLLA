from pickle import GLOBAL
import numpy as np
import csv
import os
import glob
import shutil
import random

random.seed(0)


levelfilenames = os.listdir("levelstxt/")

numlevels = len(levelfilenames)

levellen = 1000

#get max level size
for i in range(numlevels):
    level = open('levelstxt/' + levelfilenames[i], 'r')
    if levellen > len(level.readline()):
        levellen = len(level.readline())

print(levellen)
quit()
onehot = np.zeros((numlevels, 10, levellen, 17), dtype=np.int8)

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


for i in range(numlevels):
    level = open('levelstxt/' + levelfilenames[i], 'r')
    for i in range(4):
        level.readline()
    for y in range(10):
        strip = level.readline()
        for x in range(levellen):
            if strip[x] != '\n':
                if y == 9 and strip[x] == 'X':
                    onehot[i][y][x][1] = 1
                else:
                    onehot[i][y][x][symboldict[strip[x]]] = 1


np.save("../MarioPCGStudy/OriginalMario/levels", onehot)




# make "log" values:
logs = np.zeros((numlevels * levellen, 41), dtype=np.int16)

# fill log data with noise, except for output values
fillvalues = 0
for lvl in range(numlevels):
    offset = lvl * levellen
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

        # Random x
        logs[offset + x][40] = x#random.randint(0, levellen)


np.savetxt("../MarioPCGStudy/OriginalMario/logs.csv", logs, delimiter=",", fmt='%d')