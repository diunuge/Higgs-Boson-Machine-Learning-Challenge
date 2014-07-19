
# coding: utf-8

# In[1]:

import random,string,math,csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[16]:

doc_path = 'D:/Education/Semester 7/Machine Learning/Higgs ML/'


# In[48]:

#Loading csv files
training = pd.read_csv(doc_path + 'data/training.csv')
#all.astype(float)


# In[49]:

training.info()


# In[19]:

all = list(csv.reader(open(doc_path + "data/training.csv","rb"), delimiter=','))


# In[22]:

all


# In[23]:

xs = np.array([map(float, row[1:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape


# In[50]:

xs.shape


# In[47]:

xs


# In[24]:

xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))


# In[25]:

sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])


# In[26]:

weights = np.array([float(row[-2]) for row in all[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])


# In[27]:

randomPermutation = random.sample(range(len(xs)), len(xs))
numPointsTrain = int(numPoints*0.9)
numPointsValidation = numPoints - numPointsTrain

xsTrain = xs[randomPermutation[:numPointsTrain]]
xsValidation = xs[randomPermutation[numPointsTrain:]]

sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
sSelectorValidation = sSelector[randomPermutation[numPointsTrain:]]
bSelectorValidation = bSelector[randomPermutation[numPointsTrain:]]

weightsTrain = weights[randomPermutation[:numPointsTrain]]
weightsValidation = weights[randomPermutation[numPointsTrain:]]

sumWeightsTrain = np.sum(weightsTrain)
sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])


# In[28]:

xsTrainTranspose = xsTrain.transpose()


# In[29]:

weightsBalancedTrain = np.array([0.5 * weightsTrain[i]/sumSWeightsTrain
                                 if sSelectorTrain[i]
                                 else 0.5 * weightsTrain[i]/sumBWeightsTrain\
                                 for i in range(numPointsTrain)])


# In[30]:

numBins = 10


# In[31]:

logPs = np.empty([numFeatures, numBins])
binMaxs = np.empty([numFeatures, numBins])
binIndexes = np.array(range(0, numPointsTrain+1, numPointsTrain/numBins))


# In[32]:

for fI in range(numFeatures):
    # index permutation of sorted feature column
    indexes = xsTrainTranspose[fI].argsort()

    for bI in range(numBins):
        # upper bin limits
        binMaxs[fI, bI] = xsTrainTranspose[fI, indexes[binIndexes[bI+1]-1]]
        # training indices of points in a bin
        indexesInBin = indexes[binIndexes[bI]:binIndexes[bI+1]]
        # sum of signal weights in bin
        wS = np.sum(weightsBalancedTrain[indexesInBin]
                    [sSelectorTrain[indexesInBin]])
        # sum of background weights in bin
        wB = np.sum(weightsBalancedTrain[indexesInBin]
                    [bSelectorTrain[indexesInBin]])
        # log probability of being a signal in the bin
        logPs[fI, bI] = math.log(wS/(wS+wB))


# In[33]:

def score(x):
    logP = 0
    for fI in range(numFeatures):
        bI = 0
        # linear search for the bin index of the fIth feature
        # of the signal
        while bI < len(binMaxs[fI]) - 1 and x[fI] > binMaxs[fI, bI]:
            bI += 1
        logP += logPs[fI, bI] - math.log(0.5)
    return logP


# In[34]:

def AMS(s,b):
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))


# In[35]:

validationScores = np.array([score(x) for x in xsValidation])


# In[36]:

tIIs = validationScores.argsort()


# In[37]:

wFactor = 1.* numPoints / numPointsValidation


# In[38]:

s = np.sum(weightsValidation[sSelectorValidation])
b = np.sum(weightsValidation[bSelectorValidation])


# In[39]:

amss = np.empty([len(tIIs)])


# In[40]:

amsMax = 0
threshold = 0.0


# In[41]:

for tI in range(len(tIIs)):
    # don't forget to renormalize the weights to the same sum 
    # as in the complete training set
    amss[tI] = AMS(max(0,s * wFactor),max(0,b * wFactor))
    if amss[tI] > amsMax:
        amsMax = amss[tI]
        threshold = validationScores[tIIs[tI]]
        #print tI,threshold
    if sSelectorValidation[tIIs[tI]]:
        s -= weightsValidation[tIIs[tI]]
    else:
        b -= weightsValidation[tIIs[tI]]


# In[42]:

#2.0981127820868344
amsMax


# In[43]:

#-0.27902225726337126
threshold


# In[44]:

#[<matplotlib.lines.Line2D at 0x123603490>]
plt.plot(amss)


# In[51]:

test = list(csv.reader(open(doc_path + "data/test.csv", "rb"),delimiter=','))
xsTest = np.array([map(float, row[1:]) for row in test[1:]])


# In[52]:

testIds = np.array([int(row[0]) for row in test[1:]])


# In[53]:

testScores = np.array([score(x) for x in xsTest])


# In[54]:

testInversePermutation = testScores.argsort()


# In[55]:

testPermutation = list(testInversePermutation)
for tI,tII in zip(range(len(testInversePermutation)),
                  testInversePermutation):
    testPermutation[tII] = tI


# In[56]:

submission = np.array([[str(testIds[tI]),str(testPermutation[tI]+1),
                       's' if testScores[tI] >= threshold else 'b'] 
            for tI in range(len(testIds))])


# In[57]:

submission = np.append([['EventId','RankOrder','Class']],
                        submission, axis=0)


# In[58]:

np.savetxt(doc_path + "submission.csv",submission,fmt='%s',delimiter=',')


# In[ ]:



