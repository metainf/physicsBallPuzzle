from functools import partial
import multiprocessing
import time

import numpy as np

import phyre

import ImgToObj

def circle_bounding_box(circle):
  x = circle[0]
  y = circle[1]
  r = circle[2]
  return [(x-r,y+r),(x+r,y+r),(x+r,y-r),(x-r,y-r)]

def polygon_bounding_box(polygon):
  min_x = np.amin(polygon[:,0])
  max_x = np.amax(polygon[:,0])

  min_y = np.amin(polygon[:,1])
  max_y = np.amax(polygon[:,1])

  return [(min_x,max_y),(max_x,max_y),(max_x,min_y),(min_x,min_y)]

def rect_intersect(rect1,rect2):
  l1 = rect1[0]
  r1 = rect1[2]
  l2 = rect2[0]
  r2 = rect2[2]
  # If one rectangle is on left side of other
  if(l1[0] > r2[0] or l2[0] > r1[0]):
      return False

  # If one rectangle is above other
  if(l1[1] < r2[1] or l2[1] < r1[1]):
      return False

  return True

def count_good_actions(args,tier):
  (task_id,initial_scene) = args
  cache = phyre.get_default_100k_cache(tier)
  statuses = cache.load_simulation_states(task_id)
  initial_scene_data = ImgToObj.getObjectAndGoalSequence([initial_scene])
  
  object_centroid = [0,0]
  object_bb = None
  object_data = initial_scene_data['object'][0]
  if object_data['type'] == 'polygon':
    object_centroid = np.mean(object_data['data'].astype(float),axis=0)
    object_bb = polygon_bounding_box(object_data['data'].astype(float))
  elif object_data['type'] == 'circle':
    object_centroid = object_data['data'][0:2]
    object_bb = circle_bounding_box(object_data['data'])

  goal_centroid = [0,0]
  goal_data = initial_scene_data['goal'][0]
  if goal_data['type'] == 'polygon':
    goal_centroid = np.mean(goal_data['data'].astype(float),axis=0)
  elif goal_data['type'] == 'circle':
    goal_centroid = goal_data['data'][0:2]
  
  discrete_actions = cache.action_array.tolist()
  good_action_count = 0
  solved_action_count = 0
  for action_id, test_action in enumerate(discrete_actions):
    x,y,r = ImgToObj.phyreActionToPixelAction(test_action)
    if (goal_centroid[0] - object_centroid[0]) * (object_centroid[0] - x) > 0 and statuses[action_id] != phyre.simulation_cache.INVALID:
      test_action_bb = [(x-r,y+r),(x+r,y+r),(x+r,0),(x-r,0)]
      if rect_intersect(object_bb,test_action_bb):
        good_action_count += 1
        if statuses[action_id] == phyre.simulation_cache.SOLVED:
          solved_action_count += 1
  return {'num_good':good_action_count,'num_solved':solved_action_count,'num_total':len(discrete_actions)}
  

tier = 'ball'
cache = phyre.get_default_100k_cache(tier)
task_ids = list(cache.task_ids)
simulator = phyre.initialize_simulator(task_ids, tier)
initial_scenes = simulator.initial_scenes

print(len(task_ids))

pool = multiprocessing.Pool(4)
partial_worker = partial(
    count_good_actions,
    tier=tier)

f = open("simple_agent_stats.txt", "w+")

t0 = time.time()
results = pool.imap(partial_worker,zip(task_ids,initial_scenes))
reduction_count = []
percent_solved = []
no_good_count = 0


for result in results:
  if result['num_good'] != 0:
    reduction_count.append(result['num_good'])
    percent_solved.append(result['num_solved'] / result['num_good'])
  else:
    no_good_count += 1
t1 = time.time()
print((t1-t0)/len(task_ids),"Avg Time")

print("Reduction Mean:", np.mean(reduction_count), "STD:", np.std(reduction_count), file=f)
print("Percent Solved Mean:", np.mean(percent_solved), "STD:", np.std(percent_solved), file=f)
print("Percent With No Good action:", no_good_count/len(task_ids), file=f)
