from pickle import GLOBAL
import numpy as np
import csv
import os
import shutil

# Converts the levels.csv files into a single one-hot array.
# The four indices represent Level number, X position, Y position, and tile type, respectively.


# Extracts level information for all levels.
def pickle_levels(levels):
    level_inds = {'TestLevelC0.csv': 0, 'TestLevelB11.csv': 1, 'MarioLevel.csv': 2, 'TestLevelC13.csv': 3,
    'TestLevelB1.csv': 4, 'TestLevelC5.csv': 5, 'TestLevelC16.csv': 6, 'TestLevelB13.csv': 7,
    'TestLevelC14.csv': 8, 'TestLevelB9.csv': 9, 'TestLevelB8.csv': 10, 'TestLevelD6.csv': 11,
    'TestLevelD8.csv': 12, 'TestLevelD10.csv': 13, 'TestLevelD2.csv': 14, 'TestLevelD1.csv': 15}

    # Only records first [maxlen] tiles in a level, which is the shortest level in the set.
    maxlen = 198
    # Output array. The four indices represent Level number, X position, Y position, and tile type, respectively.
    onehot = np.zeros((16, 10, maxlen, 17), dtype=np.int8)

    # record data for each level
    for levelname in levels:
        # Get level information from file.
        file = csv.reader(open('../MarioPCG/LevelParser/Levels/' + levelname))
        level_data = list(file)

        # Retrieve level index
        level_ind = level_inds[levelname]

        # Remove empty entries in the files.
        pos = 0
        for tile in level_data:
            if len(tile) == 0:
                level_data.pop(pos)
            pos+=1

        # Record all tiles, ignoring the top 3 rows and the bottom row, and quitting after reaching maxlen.
        for tile in level_data:
            if int(tile[1]) > 3 and int(tile[1]) < 14 and int(tile[0]) < maxlen:
                onehot[level_ind][int(tile[1]) - 4][int(tile[0])][int(tile[2])] = 1

    # For positions with no block, set the position's "air" value to 1
    for level in range(onehot.shape[0]):
        for y in range(onehot.shape[1]):
            for x in range(onehot.shape[2]):
                if sum(onehot[level][y][x]) == 0:
                    onehot[level][y][x][0] = 1

    # Save the output
    np.save(output_folder + "onehot", onehot)
    return


# Main function call.
def extract_all():
    # Deletes old files and remakes the directory.
    shutil.rmtree(output_folder)
    os.mkdir(output_folder)

    java_levels = os.listdir('../MarioPCG/LevelParser/Levels/')
    pickle_levels(java_levels)


output_folder = "MarioPCG/Levels/"


extract_all()