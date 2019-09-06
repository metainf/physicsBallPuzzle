import glob
import os


collisionList = glob.glob("*C.npz")
noCollisionList = glob.glob("*N.npz")
num = -1

print(len(collisionList))
print(len(noCollisionList))


for filename in collisionList:
    num += 1
    os.rename(filename,"{}C.npz".format(num))


for filename in noCollisionList:
    num += 1
    os.rename(filename,"{}N.npz".format(num))