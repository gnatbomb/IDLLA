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
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# set random seeds
random_seed = 0
tf.random.set_seed(random_seed)
random.seed(random_seed)

# Set to change what metric we are using. Use metricnames for reference for values.
metric = 2
metricnames = {
    0: "models/fun/model.tflearn",
    1: "models/frustration/model.tflearn",
    2: "models/challenge/model.tflearn",
    3: "models/design/model.tflearn"
}





print("Loading logs...")
# fun = 31, frust = 32, challenge = 33, design = 34
logs, labels = load_csv('output/results.csv', target_column=31+metric, categorical_labels=True, n_classes=3)
print("Loading complete!")


# Preprocessing function
def preprocess(timesteps, columns_to_delete, frame_count, chunksize, num_symbols, levelfilename):
    print("Preprocessing...")
    
    metadata = []
    # preserve metadata
    # 'Fun', 'Frustration', 'Challenge', 'Design', 'SMBRank', 'MarioRank', 'PlatformerRank', 'GamesRank', 'Level_ind', mariox
    for ts in range(len(timesteps)):
        meta = []
        for col in range(34, 38):
            meta.append(timesteps[ts][col])
        metadata.append(meta)

    # get frame data
    # levels[level][y][x][symbol]
    levels = np.load(levelfilename)


    #for y in range(10):
        #oo = []
        #for x in range(10):
            #oo.append(levels[0][y][x][0])
        #print(oo)

    flatlevels = levels.flatten()


    minx = 0#math.ceil(chunksize / 2)
    maxx = levels.shape[2] - minx - 1

    chunklen = chunksize * chunksize * num_symbols
    levelmult = levels.shape[1] * levels.shape[2] * levels.shape[3]
    marioxmult = levels.shape[3]

    #chunks = np.zeros((len(timesteps), chunksize, chunksize, num_symbols), dtype=np.int8)
    chunks = np.zeros((len(timesteps) - frame_count, 3, chunklen), dtype=np.int8)

    

    for ts in range(len(timesteps) - frame_count):
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
            #chunks[ts] = levels[level][0:chunksize][mariox - minx : mariox + minx]
            #chunkbuild = np.zeros((chunksize, chunklen), dtype=np.int8)
            #for i in range(chunksize):
                #chunkbuild[i] = flatlevels[chunkstart + (newymult * i) :chunkstart + (newymult * i) + chunklen]
            
            chunks[ts][step] = flatlevels[chunkstart:chunkstart + chunklen]
        
        #for x in range(chunksize):
        #    for y in range(chunksize):
        #        for sym in range(num_symbols):
        #            metadata[ts].append(levels[level][y][x][sym])


    # Sort by descending id and delete columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [timestep.pop(column_to_delete) for timestep in timesteps]



    #check multi-chunks
    for ts in range(len(timesteps)):
        #for frame in range(1, frame_count):
        #    timesteps[ts].extend(timesteps[ts + frame])
        timesteps[ts].extend(metadata[ts])

    event_count = len(timesteps[0])
    ts_count = len(timesteps)
    
    final_array = np.zeros((ts_count - frame_count, frame_count, event_count), dtype=np.int8)
    for ts in range(ts_count - frame_count - 1):
        if ts % 50000 == 0:
            print("Processing timestep " + str(ts) + " / " + str(ts_count))
        final_array[ts] = timesteps[ts:ts+frame_count]
    
    print("Preprocessing complete!")
    return final_array, chunks, levels #np.array(timesteps, dtype=np.int8)

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

level_indices = {}
for key in level_names.keys():
    level_indices[level_names[key]] = key

# Ignore 'Fun' = 31, 'Frustration' = 32, 'Challenge' = 33, and 'Design' = 34 columns
# reminder: The target column is removed, so each one after that is 1 less.
to_ignore=[]
for i in range(31, 38):
    to_ignore.append(i)

# Logs stuff
frame_count = 10
chunksize = 10
num_symbols = 17
chunktotalsize = chunksize * chunksize * num_symbols


# Model stuff
conv_size = 5
kernels = 8
dropout = 0.98
poolsize = (3, 3)
epochs = 30
learning_rate = 0.00007
batch_size = 32
folds = 10




# Run types.
output_explanation = False  # Outputs maximally-activated filters
predict_gwario = False       # Predict gwario stuff
predict_original = True
randomize_gwario = True     # Whether to randomize the level flags for gwario data
trainfolds = False


# Preprocess data
logs, chunks, levels = preprocess(logs, to_ignore, frame_count, chunksize, num_symbols, "LevelsTxt/onehot.npy")

# Flatten and append logs and chunks
flatchunks = np.reshape(chunks, [-1, 3 * chunktotalsize])
flatlogs = np.reshape(logs, [-1, frame_count * logs.shape[2]])
final_input = np.concatenate((flatlogs, flatchunks), axis=1)

foldsize = final_input.shape[0] // folds



    

#final_input = flatchunks

# Build neural network
print("Building model...")


# Input
inputlayer = tflearn.input_data(shape=[None, final_input.shape[1]]) 

logsflat = tf.slice(inputlayer, [0,0],[-1, flatlogs.shape[1]])
chunksflat = tf.slice(inputlayer, [0, flatlogs.shape[1]], [-1, flatchunks.shape[1]])

logsshape = tf.reshape(logsflat, [-1, frame_count, logs.shape[2]])
print("logsshape = " + str(logsshape))
chunksshape = tf.reshape(chunksflat, [-1, chunksize, chunksize, num_symbols])
print("chunksshape = " + str(chunksshape))


# LOGS STUFF
# Convolution 8 kernels (5 * Event count)
logsconv1 = tflearn.conv_1d(logsshape, kernels, conv_size, logs.shape[2], activation='relu')
print("Logsconv1shape = " + str(logsconv1))

# Layer
#net = tflearn.fully_connected(net, len(logs[0][0]), activation='relu',)

# max-pool 2x2 kernels
logsmax1 = tflearn.max_pool_1d(logsconv1, 2, kernels)
print("logsmaxshape = " + str(logsmax1))

# Convolution 8 kernels (len(logs[0][0]) * frame_count)
logsconv2 = tflearn.conv_1d(logsmax1, kernels * 2, (conv_size - 2, len(logs[0][0])), activation='relu')
print("Logsconv2shape = " + str(logsconv2))
# LSTM
#logsnet = tflearn.lstm(logsnet, 64, dropout=dropout, activation='relu', return_seq=True)
#logsnet = tflearn.lstm(logsnet, 64, dropout=dropout, activation='relu', return_seq=True)

# layer
#net = tflearn.fully_connected(net, len(logs[0][0]) * poolsize[0] * poolsize[1], activation='relu')

# nother layer
#net = tflearn.fully_connected(net, len(logs[0][0]), activation='relu')


# CHUNKS STUFF
print("Chunksshape = " + str(chunksshape))
chunksconv1 = tflearn.conv_2d(chunksshape, kernels, (conv_size, conv_size), activation='relu')
print("chunksconv1 = " + str(chunksconv1))

chunksmax1 = tflearn.max_pool_2d(chunksconv1, (2, 2))
print("chunksmax1 = " + str(chunksmax1))

chunksconv2 = tflearn.conv_2d(chunksmax1, kernels, (conv_size, conv_size), activation='relu')
print("chunksconv2 = " + str(chunksconv2))

# Mix it all together
logsnet = tf.reshape(logsconv2, [-1, logsconv2.shape[1] * logsconv2.shape[2]])
chunksnet = tf.reshape(chunksconv2, [-1, 3 * chunksconv2.shape[1] * chunksconv2.shape[2] * chunksconv2.shape[3]])
print("Chunksnet = " + str(chunksnet))
finalnet = tf.concat([logsnet, chunksnet], 1)
print("finalnet = " + str(finalnet))

#finalnet = chunksnet

# Dropout
finalnetdropout = tflearn.dropout(finalnet, dropout)


finalnetfullyconnected = tflearn.fully_connected(finalnetdropout, final_input.shape[1], activation='relu')
print("finalnetfullyconnected = " + str(finalnetfullyconnected))
#Faster version, previous version as of before july 12
#finalnetfullyconnected = tflearn.fully_connected(finalnetdropout, 128, activation='relu')

# Output layer
finalnetout = tflearn.fully_connected(finalnetfullyconnected, 3, activation='softmax')
print("finalnetout = " + str(finalnetout))
finalnetregression = tflearn.regression(finalnetout, learning_rate=learning_rate)
# Define model
model = tflearn.DNN(finalnetregression)


# Find max activating chunks
if output_explanation:
    model.load(metricnames[metric])
    weights = np.array(model.get_weights(chunksconv1.W))

    levelscores = np.zeros((levels.shape[0], levels.shape[1] - weights.shape[0], levels.shape[2] - weights.shape[1], weights.shape[3]))

    # change weights for faster matrix multiplication
    newweights = np.zeros((weights.shape[3], weights.shape[0], weights.shape[1], weights.shape[2]))

    for filter in range(weights.shape[3]):
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                for symbol in range(weights.shape[2]):
                    newweights[filter][y][x][symbol] = weights[y][x][symbol][filter]

    flevels = np.zeros((levels.shape[0], levels.shape[1] - weights.shape[0], levels.shape[2] - weights.shape[0], ))

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

levels = 5

if predict_gwario or predict_original:
    model.load(metricnames[metric])

    # get gwario logs
    if predict_original:
        logs, labels = load_csv('OriginalMario/logs.csv', target_column=31+metric, categorical_labels=True, n_classes=3) 
    else:
        logs, labels = load_csv('gwario/logs.csv', target_column=31+metric, categorical_labels=True, n_classes=3)   

    # Preprocess data
    if predict_original:
        logs, chunks, levels = preprocess(logs, to_ignore, frame_count, chunksize, num_symbols, "OriginalMario/levels.npy")
    else:
        logs, chunks, levels = preprocess(logs, to_ignore, frame_count, chunksize, num_symbols, "gwario/GwarioLevels.npy")

    # Randomize the level flag
    if randomize_gwario:
        for i in range(logs.shape[0]):
            for j in range(logs.shape[1]):
                logs[i][j][-2] = random.randint(0, 15)

    flatchunks = np.reshape(chunks, [-1, 3 * chunktotalsize])

    flatlogs = np.reshape(logs, [-1, frame_count * logs.shape[2]])

    final_input = np.concatenate((flatlogs, flatchunks), axis=1)

    #final_input = np.reshape(chunks, [-1, 3 * chunktotalsize])

    '''
    # predict
    preds = model.predict(chunksflat)

    predvals = np.zeros(preds.shape[0], dtype=np.int8)

    for ts in range(preds.shape[0]):
        maxpred = 0
        maxrate = preds[ts][0]
        if maxrate < preds[ts][1]:
            maxpred = 1
            maxrate = preds[ts][1]
        if maxrate < preds[ts][2]:
            maxpred = 2
        
        predvals[ts] = maxpred



    quit()'''
trainfolds = True
# k folds training
if trainfolds and not (predict_gwario or predict_original):
    #shuffleinput = np.zeros((final_input.shape[0] + 1, final_input.shape[1]+3), dtype=np.int8)
    #for i in range(final_input.shape[0]):
    #    shuffleinput[i][0:-3] = final_input[i]
    #    shuffleinput[i][-3:] = labels[i]
    #np.random.shuffle(shuffleinput)
    #for i in range(final_input.shape[0]):
    #    final_input[i] = shuffleinput[i][:-3]
    #    labels[i] = shuffleinput[i][-3:]
    #shuffleinput = 0

    out = np.zeros(folds)
    standin = np.zeros((foldsize, final_input.shape[1]), dtype=np.int8)
    standinlabels = np.zeros((foldsize, labels.shape[1]), dtype=np.float64)

    testsetinds = np.zeros((foldsize), dtype=np.int8)
    #testset = np.zeros((foldsize, final_input.shape[1]), dtype=np.int8)
    #testsetlabels = np.zeros((foldsize, labels.shape[1]), dtype=np.float64)
    testset = np.zeros((final_input.shape[0]//100, final_input.shape[1]), dtype=np.int8)
    testsetlabels = np.zeros((final_input.shape[0]//100, labels.shape[1]), dtype=np.float64)
    #keepindex = foldsize
    #testset = final_input[-keepindex:]
    #testsetlabels = labels[-(keepindex+10):-10]

    # For data points taken randomly throughout the set.
    keeprate = 0.15
    keepindex = 0
    for i in range(1, final_input.shape[0]):
        #if keepindex < foldsize and random.random() < keeprate:
        if i % 90 == 0 and keepindex < final_input.shape[0]//100: # For testing if test data is being included indirectly in training set
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
    #for i in range(keepindex):
    #    r = testsetinds[i]
        #final_input[r] = final_input[-i]
        #labels[r] = labels[(-i-10)]
    
    #final_input[-keepindex:] = testset
    #labels[-keepindex:-10] = testsetlabels'''


    for f in range(1):
        #train
        model.fit(newfinalinput[0:endindex], newlabels[0:endindex], n_epoch=epochs, batch_size=batch_size, show_metric=True) # test set test
        #model.fit(final_input[0:-keepindex], labels[0:-(keepindex + 10)], n_epoch=epochs, batch_size=batch_size, show_metric=True)
        #fold_input = 0
        #fold_actuals = 0



        #output
        #fold_testset = final_input[(folds-1)*foldsize:(folds)*foldsize]
        preds = model.predict(testset)
        #fold_testset = 0

        #actuals = labels[(folds-1)*foldsize:(folds)*foldsize]
        actuals = testsetlabels

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

        #standin = final_input[0:foldsize]
        #standinlabels = labels[0:foldsize]
        #for section in range(folds - 1):
        #    final_input[section*foldsize:(section+1)*foldsize] = final_input[(section+1)*foldsize:(section+2)*foldsize]
        #    labels[section*foldsize:(section+1)*foldsize] = labels[(section+1)*foldsize:(section+2)*foldsize]
        #final_input[(folds-1)*foldsize:] = standin
        #labels[(folds-1)*foldsize:-10] = standinlabels
        
    
    #print(out)
    #print("Total Acc: " + str(sum(out) / folds))
    quit()



# Start training (apply gradient descent algorithm)
if not (predict_gwario or predict_original):
    print("Training...")
    model.fit(final_input, labels, n_epoch=epochs, batch_size=batch_size, show_metric=True)




# get prediction set
pred_freq = 0.1
if predict_gwario or predict_original:
    pred_freq = 1.0
pred_list = []
actuals = []
for frame in range(len(final_input)):
    if random.random() <= pred_freq:
        pred_list.append(final_input[frame])
        actuals.append(labels[frame])



pred_array = np.array(pred_list, dtype=np.int8)
preds = model.predict(pred_array)

if predict_original:
    #levels = np.load("OriginalMario/levels.npy") #unnecesary
    # Here we output overall values instead of strict values.
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


if predict_gwario and False:
    #levels = np.load("OriginalMario/levels.npy") #unnecesary
    # Here we output overall values instead of strict values.
    confidence = np.zeros((levels.shape[0], 4))

    for lvl in range(levels.shape[0]):
        confidence[lvl][0] = lvl
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
        
        confidence[lvl][0] = (2 * confidence[lvl][1]) + (1 * confidence[lvl][2])

    print(confidence)


    quit()






actuals1d = np.zeros(len(actuals), dtype=np.int32)

for ts in range(len(actuals)):
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

    #conf_mat[0][actual] += pred[0]
    #conf_mat[1][actual] += pred[1]
    #conf_mat[2][actual] += pred[2]


#tf.math.confusion_matrix(preds1d, actuals1d, num_classes=3))
#quit()



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
total = conf_mat.sum()
conf_mat = conf_mat / total * 100


for i in range(3):
    for j in range(3):
        lefthalf = str(conf_mat[i][j]).split('.')[0]
        righthalf = str(conf_mat[i][j]).split('.')[1]
        lefthalf += "."
        lefthalf += righthalf[0:4]
        conf_mat[i][j] = float(lefthalf)

print("Total accuracy: " + str(conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]))

print(conf_mat)

print("Prediction rates:")
for i in range(0, 3):
    print(str(i) + " preds: " + str(conf_mat[i].sum()) + "    " + str(i) + " accuracy: " + str(100 * conf_mat[i][i] / conf_mat[i].sum()))

if not predict_gwario or predict_original:
    model.save(metricnames[metric])