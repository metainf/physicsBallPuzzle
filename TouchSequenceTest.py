import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

#import pymunk
#from pymunk import Vec2d

import phyre

import ImgToObj

eval_setup = 'ball_cross_template'
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

task_str = '00000:000'

task_data_dict = phyre.loader.load_compiled_task_dict()

simulator = phyre.initialize_simulator([task_str], action_tier)
#action = [.84,.82,.41]
#action = [0.8720595836408028,0.1325951705610915,0.40200105882798676]
action = [0,0,0]

t0 = time.time()
sim_result = simulator.simulate_action(0, action, need_images=True,stride=2)
t1 = time.time()
print(t1-t0,"Sim Time")

print(sim_result.status.is_solved())

t0 = time.time()
seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
t1 = time.time()
print(t1-t0,"Sequence Contour Finding Time")

t0 = time.time()
print(ImgToObj.objectTouchGoalSequence(sim_result.images))
t1 = time.time()
print(t1-t0,"Sequence Touch Finding Time")

print(len(sim_result.images))

start_img = phyre.vis.observations_to_float_rgb(sim_result.images[0])
fig, ax = plt.subplots()
ax.imshow(start_img)

patches = []

for frame_data in seq_data:
  for layer_data in frame_data:
    if layer_data[0] == "polygon":
      verts = layer_data[1]
      verts[:,1] = 256.0-verts[:,1]
      polygon = Polygon(verts,facecolor='b',edgecolor='k',fill=True,alpha=.3)
      patches.append(polygon)
    else:
      circle_data = layer_data[1]
      circle1=plt.Circle((circle_data[0],256.0-circle_data[1]),radius=circle_data[2],facecolor='g',fill=True,alpha=.3)
      ax.add_artist(circle1)

p1 = PatchCollection(patches,alpha=.3)
ax.add_collection(p1)
plt.show()
