import time

import numpy as np
from numpy.polynomial import polynomial as P

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import phyre

import ImgToObj

eval_setup = 'ball_cross_template'
tier = 'ball'
task_str = '00007:008'
stride = 5

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
ax.imshow(start_img)

patches = []
layer_names = ['object','goal']
layer_colors = ['g', 'b']

goal_type = ImgToObj.Layer.dynamic_goal.value
if goal_type not in images[0]:
  goal_type = ImgToObj.Layer.static_goal.value

for layer_color,layer_name in zip(layer_colors,layer_names):
  for frame_data in seq_data[layer_name]:
    if frame_data['type'] == "polygon":
      verts = np.copy(frame_data['data'])
      verts[:,1] = 256.0-verts[:,1]
      polygon = Polygon(verts,facecolor=layer_color,edgecolor='k',fill=True,alpha=.3)
      patches.append(polygon)

      verts = np.copy(frame_data['bb'])
      verts[:,1] = 256.0-verts[:,1]
      polygon = Polygon(verts,facecolor=layer_color,edgecolor='k',fill=False,alpha=.3)
      patches.append(polygon)
    elif frame_data['type'] == "circle":
      circle_data = np.copy(frame_data['data'])
      circle1=plt.Circle((circle_data[0],256.0-circle_data[1]),radius=circle_data[2],facecolor=layer_color,fill=True,alpha=.3)
      ax.add_artist(circle1)

      verts = np.copy(frame_data['bb'])
      verts[:,1] = 256.0-verts[:,1]
      polygon = Polygon(verts,facecolor=layer_color,edgecolor='k',fill=False,alpha=.3)
      patches.append(polygon)
    

t0 = time.time()
cache = phyre.get_default_100k_cache(tier)
statuses = cache.load_simulation_states(task_str)
t1 = time.time()
print(t1-t0,"Cache Load Time")
discrete_actions = cache.action_array.tolist()

for action_id,action in enumerate(discrete_actions): 
  if statuses[action_id] == phyre.simulation_cache.SOLVED and ImgToObj.check_seq_action_intersect(seq_data, stride, goal_type, action):
    x, y, r = ImgToObj.phyreActionToPixelAction(action)
    circle1=plt.Circle((x,256.0-y),radius=r,facecolor='y',fill=True,alpha=.3)
    ax.add_artist(circle1)
  elif statuses[action_id] == phyre.simulation_cache.SOLVED:
    x, y, r = ImgToObj.phyreActionToPixelAction(action)
    circle1=plt.Circle((x,256.0-y),radius=r,facecolor='r',fill=True,alpha=.3)
    ax.add_artist(circle1) 
  
p1 = PatchCollection(patches,alpha=.3)
ax.add_collection(p1)
plt.show()

print(len(images))
print(len(seq_data['object']))

time = np.arange(len(images)) * stride / 60
y_pos = np.array([frame_data['centroid'][1] for frame_data in seq_data['object']])

calc_time = 15

series = P.polyfit(time[:calc_time],y_pos[:calc_time],[2,0])

np.set_printoptions(suppress=True)
print(series)

fig, ax = plt.subplots()
ax.plot(time[:calc_time],y_pos[:calc_time])
ax.plot(time[:calc_time],P.polyval(time[:calc_time],series))
plt.show()