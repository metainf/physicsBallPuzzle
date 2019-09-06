import subprocess
import glob

collisionNum = 750
noCollisionNum = 250

collisionList = glob.glob("*C.npz")
print(collisionNum-len(collisionList))
for n in range(max(collisionNum-len(collisionList),0)):
    proc = subprocess.run(["python3","ballEnviroment.py","-c",str(n)])

noCollisionList = glob.glob("*N.npz")
print(noCollisionNum-len(noCollisionList))
for n in range(max(noCollisionNum-len(noCollisionList),0)):
    proc = subprocess.run(["python3","ballEnviroment.py",str(n)])
