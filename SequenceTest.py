import time
import os
import numpy as np
from numpy.polynomial import polynomial as P

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import phyre

import ImgToObj

eval_setup = 'ball_cross_template'
tier = 'ball'
task_str = '00021:002'
stride = 50

task_dict = phyre.loader.load_compiled_task_dict()

t0 = time.time()
task = task_dict[task_str]
t1 = time.time()
print(t1-t0,"Load Time")

t0 = time.time()
_, _, images,_ = phyre.simulator.magic_ponies(task, phyre.simulator.scene_if.UserInput(),need_images=True,stride = stride)
t1 = time.time()
print(t1-t0,"Sim Time")

t0 = time.time()
seq_data = ImgToObj.getObjectAndGoalSequence(images)
t1 = time.time()
print(t1-t0,"Sequence Contour Finding Time")

start_img = phyre.vis.observations_to_float_rgb(images[0])
fig, ax = plt.subplots()


goal_type = ImgToObj.Layer.dynamic_goal.value
if goal_type not in images[0]:
  goal_type = ImgToObj.Layer.static_goal.value

t0 = time.time()
cache = phyre.get_default_100k_cache(tier)
statuses = cache.load_simulation_states(task_str)
t1 = time.time()
print(t1-t0,"Cache Load Time")
discrete_actions = cache.action_array.tolist()

t0 = time.time()
ax.imshow(start_img)

for action_id,action in enumerate(discrete_actions): 
  if statuses[action_id] != phyre.simulation_cache.INVALID and ImgToObj.check_seq_action_intersect(images[0],seq_data, stride, goal_type, action):
    x, y, r = ImgToObj.phyreActionToPixelAction(action)
    circle1=plt.Circle((x,256.0-y),radius=r,facecolor='r',fill=True,alpha=.1)
    ax.add_patch(circle1)

ax.plot()

for img in images:
  alphaImg = np.zeros((256,256,4))
  alphaImg[:,:,3] = .6
  alphaImg[:,:,0:3] = phyre.vis.observations_to_float_rgb(img)
  alphaImg[:,:,3] = (1 * (alphaImg[:, :, :3] != 1).any(axis=2))
  ax.imshow(alphaImg, zorder=10)

t1 = time.time()
print(t1-t0,"Plot Load Time")
filename = "SeqAgent.png"

i = 1
while os.path.isfile(filename):
  filename = "SeqAgent{}.png".format(i)
  i += 1

plt.savefig(filename)
plt.show()