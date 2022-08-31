import chunk
from pickletools import float8
from turtle import right
from typing import final
import numpy as np
import tflearn
import random
import tensorflow as tf
import os
import math
from scipy import stats

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Disables GPU processing. Turned out to be slower than CPU on our machine.

# set random seeds
random_seed = 0
tf.random.set_seed(random_seed)
random.seed(random_seed)

# Which metric we use. Metric indices shown in metricnames.
metric = 2
metricnames = {
    0: "Models/fun/model.tflearn",
    1: "Models/frustration/model.tflearn",
    2: "Models/challenge/model.tflearn",
    3: "Models/design/model.tflearn"
}



# Ignore 'Fun' = 31, 'Frustration' = 32, 'Challenge' = 33, and 'Design' = 34 columns
# Seemingly removes the demographic information, but those are kept.
# reminder: The target column is removed, so each one after that is 1 less.
to_ignore=[]
for i in range(31, 38):
    to_ignore.append(i)

# Model parameters. Can change these to change the model.
frame_count = 10        # How many frames we should capture.
conv_size = 5           # How large we want our convolution layers to be
kernels = 8             # How many kernels we wish to use
dropout = 0.98          # Dropout layer keep rate
poolsize = (3, 3)       # Convolution pool size
epochs = 30             # Number of training epochs
learning_rate = 0.00007 # Learning rate of the model
batch_size = 32         # How long we go before updating the model
folds = 10              # Number of folds (deprecated, might remove)

# Do not change these model parameters unless heavily altering this code.
chunksize = 10      # How large each level chunk should be
num_symbols = 17    # How many block types exist.
chunktotalsize = chunksize * chunksize * num_symbols        # Used for math stuff for slicing chunks.



# Which experiment we are running.
output_max_filters = False   # Outputs maximally-activated filters
predict_gwario = False       # Predict Gwario accuracy on trained model
predict_original = False     # Predict Original mario levels challenge ratings
trainfolds = False           # Fold checking (deprecated.)

experiment = 3

if experiment == 1:
    output_max_filters = True
if experiment == 2:
    predict_gwario = True
if experiment == 3:
    predict_original = True
if experiment == 4:
    trainfolds = True



randomize_gwario = True      # Whether to randomize the level flags for gwario or original level experiments


# -------------------------------------------------------- NO CHANGES SHOULD BE NECESSARY BELOW THIS LINE -------------------------------------

print("Loading logs...")
# fun = 31, frust = 32, challenge = 33, design = 34
logs, labels = load_csv('MarioPCG/logs.csv', target_column=31+metric, categorical_labels=True, n_classes=3)
print("Loading complete!")



# Preprocessing function
# Returns formatted logs, chunks, and level information.
def preprocess(timesteps, columns_to_delete, frame_count, chunksize, num_symbols, levelfilename):
    print("Preprocessing...")

    # Get level data for getting chunks. 
    # Flattens the level array and takes a chunksize x chunksize x num_symbols sized slice from the flattened array.
    levels = np.load(levelfilename)
    flatlevels = levels.flatten()

    # Level boundaries
    minx = 0
    maxx = levels.shape[2] - minx - 1

    # Used for math operations for getting the relevant slices of the levels.
    chunklen = chunksize * chunksize * num_symbols
    levelmult = levels.shape[1] * levels.shape[2] * levels.shape[3]
    marioxmult = levels.shape[3]

    # Final chunks output array
    chunks = np.zeros((len(timesteps) - frame_count, 3, chunklen), dtype=np.int8)

    
    # For each timestep, grab the related chunk given the level number and mario x.
    for ts in range(len(timesteps) - frame_count):
        # Grabs the 3 chunks from the start, middle, and end of the frame range.
        # One at frame 0, frame (framecount / 2), and frame (framecount)
        for step in range(3):
            stepval = 0
            if step == 1:
                stepval = frame_count // 2
            elif step == 2:
                stepval = frame_count

            mariox = int(timesteps[ts + stepval][-1])
            if mariox < minx:
                mariox = minx
            elif mariox > maxx:
                mariox = maxx

            level = int(timesteps[ts][-2])
            chunkstart = (level * levelmult) + (mariox * marioxmult)
            
            # Slice the chunk from the flattened levels array
            chunks[ts][step] = flatlevels[chunkstart:chunkstart + chunklen]

    # We are now done collecting level chunks.
    # Moving on to logs now.

    # preserve player demographic information
    # For reference, 31-39 are: 'Fun', 'Frustration', 'Challenge', 'Design', 'SMBRank', 'MarioRank', 'PlatformerRank', 'GamesRank', 'Level_ind', mariox
    metadata = []
    for ts in range(len(timesteps)):
        meta = []
        for col in range(34, 38):
            meta.append(timesteps[ts][col])
        metadata.append(meta)

    # Remove ignored columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [timestep.pop(column_to_delete) for timestep in timesteps]

    # Append metadata to each frame so it only appears once per set of frames, rather than 10 times per set of frames.
    for ts in range(len(timesteps)):
        timesteps[ts].extend(metadata[ts])

    # Start making the final array for outputting.
    event_count = len(timesteps[0])
    ts_count = len(timesteps)
    
    final_array = np.zeros((ts_count - frame_count, frame_count, event_count), dtype=np.int8)
    for ts in range(ts_count - frame_count - 1):
        if ts % 50000 == 0:
            print("Processing timestep " + str(ts) + " / " + str(ts_count))
        final_array[ts] = timesteps[ts:ts+frame_count]
    
    print("Preprocessing complete!")
    return final_array, chunks, levels

# Filenames for the different levels.
level_names = {0: 'TestLevelC0.csv',
1: 'TestLevelB11.csv',
2: 'MarioLevel.csv',
3: 'TestLevelC13.csv',
4: 'TestLevelB1.csv',
5: 'TestLevelC5.csv',
6: 'TestLevelC16.csv',
7: 'TestLevelB13.csv',
8: 'TestLevelC14.csv',
9: 'TestLevelB9.csv',
10: 'TestLevelB8.csv',
11: 'TestLevelD6.csv',
12: 'TestLevelD8.csv',
13: 'TestLevelD10.csv',
14: 'TestLevelD2.csv',
15: 'TestLevelD1.csv'}

# Reverse dictionary for looking up which level is used.
level_indices = {}
for key in level_names.keys():
    level_indices[level_names[key]] = key


# Preprocess data
logs, chunks, levels = preprocess(logs, to_ignore, frame_count, chunksize, num_symbols, "levels.npy")

# --------------------------------------------vvvvv Constructing our CNN vvvvvv-----------------------------------------------------

# Flatten and append logs and chunks
flatchunks = np.reshape(chunks, [-1, 3 * chunktotalsize])
flatlogs = np.reshape(logs, [-1, frame_count * logs.shape[2]])
final_input = np.concatenate((flatlogs, flatchunks), axis=1)

foldsize = final_input.shape[0] // folds # deprecated.


# Build neural network
print("Building model...")

# Input
inputlayer = tflearn.input_data(shape=[None, final_input.shape[1]]) 

# Separate out the chunks and logs layers
logsflat = tf.slice(inputlayer, [0,0],[-1, flatlogs.shape[1]])
chunksflat = tf.slice(inputlayer, [0, flatlogs.shape[1]], [-1, flatchunks.shape[1]])

# Reshape the logs and chunks layers
logsshape = tf.reshape(logsflat, [-1, frame_count, logs.shape[2]])
print("logsshape = " + str(logsshape))
chunksshape = tf.reshape(chunksflat, [-1, chunksize, chunksize, num_symbols])
print("chunksshape = " + str(chunksshape))


# LOGS STUFF
# Convolution 8 kernels (5 * Event count)
logsconv1 = tflearn.conv_1d(logsshape, kernels, conv_size, logs.shape[2], activation='relu')
print("Logsconv1shape = " + str(logsconv1))

# max-pool 2x2 kernels
logsmax1 = tflearn.max_pool_1d(logsconv1, 2, kernels)
print("logsmaxshape = " + str(logsmax1))

# Convolution 8 kernels (len(logs[0][0]) * frame_count)
logsconv2 = tflearn.conv_1d(logsmax1, kernels * 2, (conv_size - 2, len(logs[0][0])), activation='relu')
print("Logsconv2shape = " + str(logsconv2))


# CHUNKS STUFF
print("Chunksshape = " + str(chunksshape))
chunksconv1 = tflearn.conv_2d(chunksshape, kernels, (conv_size, conv_size), activation='relu')
print("chunksconv1 = " + str(chunksconv1))

chunksmax1 = tflearn.max_pool_2d(chunksconv1, (2, 2))
print("chunksmax1 = " + str(chunksmax1))

chunksconv2 = tflearn.conv_2d(chunksmax1, kernels, (conv_size, conv_size), activation='relu')
print("chunksconv2 = " + str(chunksconv2))

# Flatten the logs and chunks portions to combine them
logsnet = tf.reshape(logsconv2, [-1, logsconv2.shape[1] * logsconv2.shape[2]])
chunksnet = tf.reshape(chunksconv2, [-1, 3 * chunksconv2.shape[1] * chunksconv2.shape[2] * chunksconv2.shape[3]])
print("Chunksnet = " + str(chunksnet))

# Combine logs and chunks.
finalnet = tf.concat([logsnet, chunksnet], 1)
print("finalnet = " + str(finalnet))

# Dropout layer.
finalnetdropout = tflearn.dropout(finalnet, dropout)

# Fully connected layer
finalnetfullyconnected = tflearn.fully_connected(finalnetdropout, final_input.shape[1], activation='relu')
print("finalnetfullyconnected = " + str(finalnetfullyconnected))

# Output layer. Output of 0 = most x rating, 1 = mid x rating, 2 = least x rating.
# Where x is the metric being predicted.
finalnetout = tflearn.fully_connected(finalnetfullyconnected, 3, activation='softmax')
print("finalnetout = " + str(finalnetout))


finalnetregression = tflearn.regression(finalnetout, learning_rate=learning_rate)
# Define model
model = tflearn.DNN(finalnetregression)

# ------------------------------------------------DONE CONSTRUCTING THE MODEL------------------------------------------

# Output maximally activated chunks.
# For each level, prints out the maximally activating chunk for each filter.
if output_max_filters:
    model.load(metricnames[metric])
    weights = np.array(model.get_weights(chunksconv1.W))

    # Activation rating for each 5x5 xy position.
    levelscores = np.zeros((levels.shape[0], levels.shape[1] - weights.shape[0], levels.shape[2] - weights.shape[1], weights.shape[3]))

    # change weights for faster matrix multiplication
    newweights = np.zeros((weights.shape[3], weights.shape[0], weights.shape[1], weights.shape[2]))

    for filter in range(weights.shape[3]):
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                for symbol in range(weights.shape[2]):
                    newweights[filter][y][x][symbol] = weights[y][x][symbol][filter]

    # For recording the parts of the levels with the highest activations
    maxsections = np.zeros((levels.shape[0], weights.shape[3], weights.shape[1], weights.shape[0]), dtype=np.int8)

    topinds = np.zeros((weights.shape[3], 3), dtype=np.int32)
    topscores = np.zeros((weights.shape[3]), dtype=np.float64)
    # For each level
    for level_ind in range(levels.shape[0]):
        print("Multiplying weights for level: " + str(level_ind))

        # Find the activations
        # For each filter
        for filternum in range(weights.shape[3]):
            # for y spread. (Currently set to catch only the single height we use.)
            for y in range(levels.shape[1] - weights.shape[0]):
                # For each x
                for x in range(levels.shape[2] - weights.shape[1]):
                    # for range of y
                    for yi in range(weights.shape[0]):
                        # for range of x
                        for xi in range(weights.shape[1]):
                            levelscores[level_ind][y][x][filternum] += np.sum(np.multiply(levels[level_ind][y+yi][x+xi], newweights[filternum][yi]))

        
        # Find the top values and positions
        for filternum in range(weights.shape[3]):
            topscore = levelscores[level_ind][0][0][filternum]
            
            topind = [0,0]
            # for y spread. (Currently set to catch only the single height we use.)
            for y in range(levels.shape[1] - weights.shape[0]):
                # For each x
                for x in range(levels.shape[2] - weights.shape[1]):
                    if levelscores[level_ind][y][x][filternum] > topscore:
                        topscore = levelscores[level_ind][y][x][filternum]
                        topind = [y, x]
                        if topscore > topscores[filternum]:
                            topscores[filternum] = topscore
                            topinds[filternum][0] = x
                            topinds[filternum][1] = y
                            topinds[filternum][2] = level_ind
            # Record the 5x5 values of the top scorer for this filter
            for y in range(weights.shape[0]):
                for x in range(weights.shape[1]):
                    for symbol in range(weights.shape[2]):
                        if levels[level_ind][y+topind[0]][x+topind[1]][symbol] != 0:
                            maxsections[level_ind][filter][y][x] = symbol

            # Maximally activated for each level.
            print(maxsections[level_ind][filter])
    # Maximally activated ones
    littlemaxsections = np.zeros((weights.shape[3], 5, 5), dtype=np.int8)
    for filternum in range(weights.shape[3]):
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                for symbol in range(weights.shape[2]):
                    if levels[topinds[filternum][2]][y+topinds[filternum][1]][x+topinds[filternum][0]][symbol] != 0:
                        littlemaxsections[filter][y][x] = symbol
    quit()


# Prepares data for performing gwario or original dataset experiments
if predict_gwario or predict_original:
    model.load(metricnames[metric])

    # get relevant logs and preprocess the data
    if predict_original:
        logs, labels = load_csv('OriginalMario/logs.csv', target_column=31+metric, categorical_labels=True, n_classes=3) 
        logs, chunks, levels = preprocess(logs, to_ignore, frame_count, chunksize, num_symbols, "OriginalMario/levels.npy")
    else:
        logs, labels = load_csv('Gwario/logs.csv', target_column=31+metric, categorical_labels=True, n_classes=3)   
        logs, chunks, levels = preprocess(logs, to_ignore, frame_count, chunksize, num_symbols, "Gwario/levels.npy")
        

    # Randomize the level flag
    if randomize_gwario:
        for i in range(logs.shape[0]):
            for j in range(logs.shape[1]):
                logs[i][j][-2] = random.randint(0, 15)

    # These are used later.
    flatchunks = np.reshape(chunks, [-1, 3 * chunktotalsize])
    flatlogs = np.reshape(logs, [-1, frame_count * logs.shape[2]])
    final_input = np.concatenate((flatlogs, flatchunks), axis=1)



# Code used to verify that data points weren't being included indirectly in the verification set.
# Not presented as part of our publication.
if trainfolds:
    out = np.zeros(folds)
    standin = np.zeros((foldsize, final_input.shape[1]), dtype=np.int8)
    standinlabels = np.zeros((foldsize, labels.shape[1]), dtype=np.float64)

    testsetinds = np.zeros((foldsize), dtype=np.int8)

    testset = np.zeros((final_input.shape[0]//100, final_input.shape[1]), dtype=np.int8)
    testsetlabels = np.zeros((final_input.shape[0]//100, labels.shape[1]), dtype=np.float64)


    # For data points taken randomly throughout the set.
    keepindex = 0
    for i in range(1, final_input.shape[0]):
        if i % 90 == 0 and keepindex < final_input.shape[0]//100: # For checking if test data is being included indirectly in training set
            testset[keepindex] = final_input[i]
            testsetlabels[keepindex] = labels[i]
            testsetinds[keepindex] = i
            keepindex += 1

    newfinalinput = np.zeros(final_input.shape, dtype=np.int8)
    newlabels = np.zeros(labels.shape, dtype=np.int8)
    newi = 0
    for i in range(0, final_input.shape[0] // 100):
        newfinalinput[newi:newi+80] = final_input[i*100:(i*100)+80]
        newlabels[newi:newi+80] = labels[i*100:(i*100)+80]
        newi += 80
    final_input = 0
    labels = 0
    endindex = (newfinalinput.shape[0]*4) // 5


    for f in range(1):
        #train
        model.fit(newfinalinput[0:endindex], newlabels[0:endindex], n_epoch=epochs, batch_size=batch_size, show_metric=True) # test set test

        #output
        preds = model.predict(testset)

        actuals = testsetlabels

        # Code copied from below.
        actuals1d = np.zeros(actuals.shape[0], dtype=np.int8)
        for ts in range(actuals.shape[0]):
            act = actuals[ts]
            if act[0]:
                actuals1d[ts] = 0
            elif act[1]:
                actuals1d[ts] = 1
            else:
                actuals1d[ts] = 2

        conf_mat = np.zeros((3,3), dtype=np.float32)

        for ts in range(len(preds)):
            pred = preds[ts]
            actual = actuals1d[ts]
            minval = 0

            for i in range(1,3):
                if pred[minval] < pred[i]:
                    minval = i
            conf_mat[minval][actual] += 1

        total = conf_mat.sum()
        conf_mat = conf_mat / total * 100
        out[f] = conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]
        print(str(out[f]))

    quit()



# Train the model (apply gradient descent algorithm)
if not (predict_gwario or predict_original):
    print("Training...")
    model.fit(final_input, labels, n_epoch=epochs, batch_size=batch_size, show_metric=True)


# Randomly select test set. 
pred_freq = 0.1
if predict_gwario or predict_original:
    pred_freq = 1.0
pred_list = []
actuals = []
for frame in range(len(final_input)):
    if random.random() <= pred_freq:
        pred_list.append(final_input[frame])
        actuals.append(labels[frame])


# Output model predictions
pred_array = np.array(pred_list, dtype=np.int8)
preds = model.predict(pred_array)

# Outputs ratings for original mario levels.
if predict_original:
    confidence = np.zeros((levels.shape[0], 4))
    levelnumbers = [1,2,3,5,9,11,13,14,17,19,21,22,23,25,29]

    for lvl in range(levels.shape[0]):
        confidence[lvl][0] = levelnumbers[lvl]
        for x in range(0, levels.shape[2] - 11):
            confidence[lvl][1:] += preds[lvl * levels.shape[2] + x]
        confidence[lvl][1:] /= (levels.shape[2] - 11)

        for i in range(1,4):
            # Make printout nicer
            l = str(confidence[lvl][i]).split('.')[0]
            r = str(confidence[lvl][i]).split('.')[1]
            l += "."
            l += r[0:4]
            confidence[lvl][i] = float(l)

    print(confidence)
    r = stats.spearmanr(confidence)
    print(r[0])



actuals1d = np.zeros(len(actuals), dtype=np.int32)

for ts in range(len(actuals)):
    act = actuals[ts]
    if act[0]:
        actuals1d[ts] = 0
    elif act[1]:
        actuals1d[ts] = 1
    else:
        actuals1d[ts] = 2

# confusion matrix for checking if we are over/underfitting.
conf_mat = np.zeros((3,3), dtype=np.float32)

for ts in range(len(preds)):
    pred = preds[ts]
    actual = actuals1d[ts]
    maxval = 0

    # Gets highest prediction
    for i in range(1,3):
        if pred[maxval] < pred[i]:
            maxval = i
    conf_mat[maxval][actual] += 1



# Outputs run information
print("Run data:")
print("Predicted metric: " + str(31 + metric))
print("Ignored flags: " + str(to_ignore))
print("frame count: " + str(frame_count))
print("Convolution layer size: " + str(conv_size))
print("Kernel count: " + str(kernels))
print("Dropout rate: " + str(dropout))
print("Maxpool size: " + str(poolsize))
print("Epochs: " + str(epochs))
print("Batch size: " + str(batch_size))
print("Learning rate: " + str(learning_rate))

print("Results:")

# Converts confusion matrix into percentages.
total = conf_mat.sum()
conf_mat = conf_mat / total * 100

# Formats output strings for confusion matrix
for i in range(3):
    for j in range(3):
        lefthalf = str(conf_mat[i][j]).split('.')[0]
        righthalf = str(conf_mat[i][j]).split('.')[1]
        lefthalf += "."
        lefthalf += righthalf[0:4]
        conf_mat[i][j] = float(lefthalf)

print("Total accuracy: " + str(conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]))

print(conf_mat)

# Outputs prediction rates and accuracies of each of the prediction classes.
print("Prediction rates:")
for i in range(0, 3):
    print(str(i) + " preds: " + str(conf_mat[i].sum()) + "    " + str(i) + " accuracy: " + str(100 * conf_mat[i][i] / conf_mat[i].sum()))

# Saves the model for later use.
if not predict_gwario or predict_original:
    model.save(metricnames[metric])