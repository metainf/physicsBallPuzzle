import time

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import phyre

import ImgToObj


eval_setup = 'ball_cross_template'
tier = 'ball'
task_str = '00000:000'
stride = 100

t0 = time.time()
task_dict = phyre.loader.load_compiled_task_dict()
t1 = time.time()
print(t1-t0,"Load All Data Time")

t0 = time.time()
task = task_dict[task_str]
t1 = time.time()
print(t1-t0,"Load Single Task Time")

t0 = time.time()
_, _, images,_ = phyre.simulator.magic_ponies(task, phyre.simulator.scene_if.UserInput(),need_images=True,stride = stride)
t1 = time.time()
print(t1-t0,"Sim Time")

t0 = time.time()
seq_data = ImgToObj.getObjectAndGoalSequence(images)
t1 = time.time()
print(t1-t0,"Sequence Contour Finding Time")

t0 = time.time()
cache = phyre.get_default_100k_cache(tier)
statuses = cache.load_simulation_states(task_str)
t1 = time.time()
print(t1-t0,"Cache Load Time")


t0 = time.time()
good_actions = []
discrete_actions = cache.action_array.tolist()
good_action_count = 0
solved_action_count = 0

goal_type = ImgToObj.Layer.dynamic_goal.value
if goal_type not in images[0]:
  goal_type = ImgToObj.Layer.static_goal.value

for action_id, test_action in tqdm(enumerate(discrete_actions),desc='Eval Actions'):
  x, y, r = ImgToObj.phyreActionToPixelAction(test_action)
  found_intersect = False
  if statuses[action_id] != phyre.simulation_cache.INVALID:
    for frame_index, object_data, goal_data in zip(range(len(seq_data['object'])), seq_data['object'], seq_data['goal']):
      if found_intersect:
        break
      goal_bb = goal_data['bb']
      object_bb = object_data['bb']
      frame_time = frame_index * stride / ImgToObj.FRAME_PER_SEC
      y_time = max(r,y + 1.0/2.0 * ImgToObj.GRAV_PIX_PER_SEC * frame_time * frame_time)
      test_action_bb = [(x-r, y+r), (x+r, y+r), (x+r, y-r), (x-r, y-r)]
      test_action_time_bb = [(x-r, y_time+r), (x+r, y_time+r), (x+r, y_time-r), (x-r, y_time-r)]
      if ImgToObj.rect_intersect(object_bb, test_action_time_bb) or ImgToObj.rect_intersect(object_bb, test_action_time_bb):
        good_action_count += 1
        found_intersect = True
        good_actions.append([x,y,r,statuses[action_id]])
        if statuses[action_id] == phyre.simulation_cache.SOLVED:
          solved_action_count += 1
      elif goal_type == ImgToObj.Layer.dynamic_goal.value:
        if ImgToObj.rect_intersect(goal_bb, test_action_time_bb):
          good_action_count += 1
          found_intersect = True
          good_actions.append([x,y,r,statuses[action_id]])
          if statuses[action_id] == phyre.simulation_cache.SOLVED:
            solved_action_count += 1

t1 = time.time()
print(t1-t0,"Scan Actions Time")
print(good_action_count,solved_action_count,len(discrete_actions))
print(np.sum((statuses == phyre.SimulationStatus.SOLVED)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

good_actions = np.array(good_actions)

ax.scatter(good_actions[:,0],good_actions[:,1],good_actions[:,2],c = good_actions[:,3],)
plt.show()