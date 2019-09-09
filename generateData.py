import subprocess
import glob
import argparse
import re

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--regen",help="Recreate Files",action="store_true")
args = parser.parse_args()

regen = args.regen

collisionNum = 750
noCollisionNum = 250

if regen:
    collisionList = sorted_nicely(glob.glob("*C.npz"))
    for n in range(collisionNum):
        print(n)
        if n < len(collisionList):
            proc = subprocess.run(["python3","ballEnviroment.py","-c",str(n),"-r",collisionList[n]])
        else:
            proc = subprocess.run(["python3","ballEnviroment.py","-c",str(n)])

    noCollisionList = sorted_nicely(glob.glob("*N.npz"))
    for n in range(noCollisionNum):
        print(n)
        if n < len(noCollisionList):
            proc = subprocess.run(["python3","ballEnviroment.py",str(n),"-r",noCollisionList[n]])
        else:
            proc = subprocess.run(["python3","ballEnviroment.py",str(n)])
else:
    collisionList = glob.glob("*C.npz")
    print(collisionNum-len(collisionList))
    for n in range(max(collisionNum-len(collisionList),0)):
        print(n)
        proc = subprocess.run(["python3","ballEnviroment.py","-c",str(n)])

    noCollisionList = glob.glob("*N.npz")
    print(noCollisionNum-len(noCollisionList))
    for n in range(max(noCollisionNum-len(noCollisionList),0)):
        print(n)
        proc = subprocess.run(["python3","ballEnviroment.py",str(n)])
