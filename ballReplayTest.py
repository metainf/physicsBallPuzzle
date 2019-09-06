import numpy as np

loaded1 = np.load('2019_8_29_14516.npz')
loaded2 = np.load('2019_8_29_14516.npz')

print(np.linalg.norm(loaded1['ballImgArray']-loaded2['ballImgArray']))
print(np.linalg.norm(loaded1['velArray']-loaded2['velArray']))
