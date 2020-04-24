import math
import random
import time
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import phyre

import ImgToObj

eval_setup = 'ball_cross_template'
tier = 'ball'
stride = 5
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
goal = 3.0 * 60.0/stride

task_str = '00004:243'
task_dict = phyre.loader.load_compiled_task_dict()
task = task_dict[task_str]
_, _, empty_images,_ = phyre.simulator.magic_ponies(task, phyre.simulator.scene_if.UserInput(),need_images=True,stride = stride)
empty_seq_data = ImgToObj.getObjectAndGoalSequence(empty_images)

goal_type = ImgToObj.Layer.dynamic_goal.value
if goal_type not in empty_images[0]:
  goal_type = ImgToObj.Layer.static_goal.value

simulator = phyre.initialize_simulator([task_str], action_tier)

testedActionsCount = 0

tested_actions = np.array([[-1,-1,-1,1,0]])
t0 = time.time()
while testedActionsCount < 1000:
  random_action = np.random.random_sample((1,5))
  if ImgToObj.check_seq_action_intersect(empty_images[0],empty_seq_data, stride, goal_type,np.squeeze(random_action[0:3])):
    sim_result = simulator.simulate_action(0, np.squeeze(random_action[:,0:3]), need_images=True, stride=5)
    if not sim_result.status.is_invalid():
      testedActionsCount += 1
      seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
      random_action[0,3] = 1.0 - np.linalg.norm(seq_data['object'][-1]['centroid'] - seq_data['goal'][-1]['centroid']) / 256.0
      random_action[0,3] += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
      random_action[0,4] = sim_result.status.is_solved()
      tested_actions = np.concatenate((tested_actions,random_action),0)
t1 = time.time()
print(t1-t0)
tested_actions = np.delete(tested_actions,0,0)

print(np.max(tested_actions[:,3]))
print(np.min(tested_actions[:,3]))
good_actions = tested_actions[tested_actions[:,4] == 1,:]
print(good_actions)

'''
cache = phyre.get_default_100k_cache(tier)
statuses = cache.load_simulation_states(task_str)
discrete_actions = cache.action_array.tolist()


for action_id,action in enumerate(discrete_actions): 
  if statuses[action_id] == phyre.simulation_cache.SOLVED and ImgToObj.check_seq_action_intersect(empty_seq_data, stride, goal_type,action):
    random_action = np.random.random_sample((1,5))
    random_action[:,0:3] = action
    sim_result = simulator.simulate_action(0, action, need_images=True, stride=stride)
    if not sim_result.status.is_invalid():
      testedActionsCount += 1
      seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
      random_action[0,3] = 1.0 - np.linalg.norm(seq_data['object'][-1]['centroid'] - seq_data['goal'][-1]['centroid']) / 256.0
      #random_action[0,3] += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
      random_action[0,4] = sim_result.status.is_solved()
      tested_actions = np.concatenate((tested_actions,random_action),0)

good_actions = tested_actions[tested_actions[:,4] == 1,:]
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print(len(good_actions),len(tested_actions))

p = ax.scatter(tested_actions[1:,0],tested_actions[1:,1],tested_actions[1:,2],s=10,c = tested_actions[1:,3])
fig.colorbar(p)
'''
if len(good_actions) > 0:
  ax.scatter(good_actions[1:,0],good_actions[1:,1],good_actions[1:,2],s=50,c = 'r',marker='*')
  
  print(np.corrcoef(tested_actions[:,4],tested_actions[:,3]))
'''

ax.set_xlabel('X Pos')
ax.set_ylabel('Y Pos')
ax.set_zlabel('Radius')

filename = "SmoothSeq.png"

i = 1
while os.path.isfile(filename):
  filename = "SmoothSeq{}.png".format(i)
  i += 1

plt.savefig(filename)

plt.show()