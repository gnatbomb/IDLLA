from pickle import GLOBAL
import numpy as np
import csv
import os
import glob
import shutil
import random

# This file converts Gwario levels from csv format into a single one-hot array.
# It also gets calculated level ratings based on the ratings assigned to the levels.
# Finally, it creates "dummy logs" that are empty, where each log increments the x position by one.

# Set random seed
random.seed(1)

# Get results file for retrieving which levels were played by whom
resultsfile = csv.reader(open('results.csv'))
results = list(resultsfile)

# For matching level names to indices in the array.
level_inds = {
    "1-1" : 0,
    "2-2" : 1,
    "2-3" : 2,
    "3-3" : 3
}

# Get level play information. Ignores multiplayer plays.
singleplays = {}
levelplays = os.listdir('LogsFromGwario/')
for i in range(len(levelplays)):
    playstr = levelplays[i]
    # Denotes that the level play was singleplayer
    if playstr[7] == '-':
        singleplays[playstr[2:]] = 0



# Score values for each level. 
# first inds 1-1, 2-2, 2-3, 3-3
# Second inds: fun, frustration, challenge, plays
total_scores = np.zeros((4, 4))

# Calculate mean level ratings across all level plays
for subject_ind in range(1, len(results)):
    # Retrieve player rating
    data = results[subject_ind]
    subject_id = data[0]

    # Scores. We do -1 to shift the ratings from 1-5 to 0-4
    fun = int(data[22]) - 1
    frust = int(data[23]) - 1
    chall = int(data[24]) - 1

    # Record the player's ratings for the level
    for key in singleplays.keys():
        if key[0:4] == subject_id:
            level = level_inds[key[-7:-4]]
            total_scores[level][0] += fun
            total_scores[level][1] += frust
            total_scores[level][2] += chall

            # Increment total plays of a level
            total_scores[level][3] += 1

# Denotes scoring system used. If true, records one level, two mid levels, and one bottom level.
# If false, records two top levels, one mid level, and one bottom level.
bestworst = True

# For each rated metric. (Fun, Frustration, and Challenge)
for i in range(3):
    worst = -1
    secondworst = -2
    worstscore = 6
    secondworstscore = 6
    best = -1
    bestscore = -1

    for level in range(4):
        total_scores[level][i] = total_scores[level][i] / total_scores[level][3]

        if bestworst:
            if total_scores[level][i] < worstscore:
                worstscore = total_scores[level][i]
                worst = level
            if total_scores[level][i] > bestscore:
                bestscore = total_scores[level][i]
                best = level
            total_scores[level][i] = 1
        else: 
            if total_scores[level][i] < worstscore:
                secondworstscore = worstscore
                secondworst = worst
                worstscore = total_scores[level][i]
                worst = level
            elif total_scores[level][i] < secondworstscore:
                secondworstscore = total_scores[level][i]
                secondworst = level
            total_scores[level][i] = 0
    
    # Record final values
    total_scores[worst][i] = 2
    if bestworst:
        total_scores[best][i] = 0
    else:
        total_scores[secondworst][i] = 1



# Now, onto getting the level layouts.
level_files = os.listdir('levels_readable/csv/')

# Find shortest level length in the set. We crop each level's length to the shortest level in the set.
shortestlen = 500
locallen = 0

for i in range(4):
    file = csv.reader(open("levels_readable/csv/" + level_files[i]))
    level_data = list(file)
    locallen = 0
    for j in range(len(level_data)):
        if int(level_data[j][0]) > locallen:
            locallen = int(level_data[j][0])
        
    if locallen < shortestlen:
        shortestlen = locallen

# Creates level output array
onehot = np.zeros((4, 10, shortestlen, 17), dtype=np.int8)

# For each level, set all of the block positions
for lvl in range(4):
    file = csv.reader(open("levels_readable/csv/" + level_files[i]))
    level_data = list(file)

    for entry_i in range(len(level_data)):
        entry = level_data[entry_i]
        if int(entry[0]) < shortestlen and int(entry[1]) < 10 and int(entry[1]) > 0:
            onehot[lvl][10 - int(entry[1])][int(entry[0])][int(entry[2])] = 1

# For each empty block position, set the "air" value to 1
for lvl in range(4):
    for y in range(10):
        for x in range(shortestlen):
            is_air = True
            for value in range(17):
                if onehot[lvl][y][x][value] != 0:
                    is_air = False
            if is_air:
                onehot[lvl][y][x][0] = 1

# Save the level array
np.save("../MarioPCGStudy/gwario/GwarioLevels", onehot)


# Finally we make the "dummy log" values. 
# Output dummy log array.
logs = np.zeros((4 * shortestlen, 41), dtype=np.int16)

# If 1, set log values randomly, if 0, leave logs empty. For our study we left the logs empty.
fillvalues = 0

# Set the log values
for lvl in range(4):
    offset = lvl * shortestlen
    for x in range(shortestlen):
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
        logs[offset + x][31] = total_scores[lvl][0]
        logs[offset + x][32] = total_scores[lvl][1]
        logs[offset + x][33] = total_scores[lvl][2]

        # Demographics
        for demographic in range(35, 39):
            logs[offset + x][demographic] = random.randint(0, 5) * fillvalues

        # Level
        logs[offset + x][39] = lvl

        # X position
        logs[offset + x][40] = x

# Save dummy logs to output.
np.savetxt("../MarioPCGStudy/gwario/logs.csv", logs, delimiter=",", fmt='%d')