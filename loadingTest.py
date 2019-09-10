import numpy as np
from numpy.lib import stride_tricks
import time
start = time.time()

numFrames=4
numVel=20
sequenceIndex = 0 + numFrames - 1
sequenceLength = 20

loaded = np.load("0C.npz")
imageSequence = np.zeros((sequenceLength,128,128,numFrames*3))
predictSequence = np.zeros((sequenceLength,2,numVel))
test = loaded['ballImgArray']
test2 = loaded['velArray']
for i in range(sequenceLength):
    imageSequence[i,:,:,:] = test[:,:,:,sequenceIndex-numFrames+1:sequenceIndex+1].reshape((128,128,3*numFrames))
    predictSequence[i,:,:] = test2[:,sequenceIndex:sequenceIndex+numVel]

print(time.time()-start)