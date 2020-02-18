from functools import partial
import multiprocessing
import time
from datetime import datetime

from tqdm import tqdm

import numpy as np

import phyre

import ImgToObj


def chunkify(lst, n):
  return [lst[i::n] for i in range(n)]


def rect_intersect(rect1, rect2):
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

def overlappingArea(rect1, rect2): 
  if rect_intersect(rect1,rect2):
    l1 = rect1[0]
    r1 = rect1[2]
    l2 = rect2[0]
    r2 = rect2[2]
    
    areaI = (min(r1[0], r2[0]) - max(l1[0], l2[0])) * (min(l1[1], l2[1]) - max(r1[1], r2[1])) 
    return areaI
  else:
    return 0

def rectArea(rect):
  l = rect[0]
  r = rect[2]
  area = abs(l[0] - r[0]) * abs(l[1] - r[1])
  return area

def count_good_actions(tasks, tier):
  simulator = phyre.initialize_simulator(tasks, tier)
  results = []
  ball_sizes = np.linspace(0.25, 1, 3)
  
  '''
  pos = np.linspace(0, 1, 25)

  actions = np.array(np.meshgrid(pos, pos, ball_sizes)).T.reshape(-1, 3)

  valid_actions = []

  for action in actions:
    x, y, r = ImgToObj.phyreActionToPixelAction(action)
    action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, y-r), (x-r, y-r)])
    if np.min(action_bb[:,0]) >= 0 and np.max(action_bb[:,0]) <= 255 and np.min(action_bb[:,1]) >= 0 and np.max(action_bb[:,1]) <= 255:
      valid_actions.append(action)

  actions = np.array(valid_actions)
  '''

  for task_index in tqdm(range(len(tasks)), desc='Evaluate tasks'):
    initial_scene = simulator.initial_scenes[task_index]
    frame_data = ImgToObj.getObjectAndGoalSequence([initial_scene])
    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in initial_scene:
      goal_type = ImgToObj.Layer.static_goal.value

    goal_data = frame_data['goal'][0]
    object_data = frame_data['object'][0]

    goal_bb = goal_data['bb']
    goal_center = goal_data['centroid']
    object_bb = object_data['bb']
    object_center = object_data['centroid']

    good_action_count = 0
    solved_action_count = 0

    actions = []
    for ball_size in ball_sizes:
      left_bound_obj = min(object_bb[:,0]) - ImgToObj.phyreRadiusToPixelRadius(ball_size)
      right_bound_obj = max(object_bb[:,0]) + ImgToObj.phyreRadiusToPixelRadius(ball_size)
      left_bound_goal = min(goal_bb[:,0]) - ImgToObj.phyreRadiusToPixelRadius(ball_size)
      right_bound_goal = max(goal_bb[:,0]) + ImgToObj.phyreRadiusToPixelRadius(ball_size)
      
      
      if goal_center[0] < object_center[0]:
        left_bound_obj = object_center[0]
        right_bound_goal = goal_center[0]
      '''
      elif goal_center[0] > object_center[0]:
        right_bound_obj = object_center[0]
        left_bound_goal = goal_center[0]
      ''' 
      x_obj = np.linspace(left_bound_obj/256,right_bound_obj/256,15)
      x_goal = np.linspace(left_bound_goal/256,right_bound_goal/256,15)
      #y_obj = np.linspace(object_center[1]/256 + ImgToObj.phyreRadiusToPixelRadius(ball_size)/256,1,15)
      #y_goal = np.linspace(goal_center[1]/256 + ImgToObj.phyreRadiusToPixelRadius(ball_size)/256,1,15)
      y_obj = np.linspace(max(object_bb[:,1])/256 + ImgToObj.phyreRadiusToPixelRadius(ball_size)/256,1,25)
      y_goal = np.linspace(max(goal_bb[:,1])/256 + ImgToObj.phyreRadiusToPixelRadius(ball_size)/256,1,25)
      actions.append(np.array(np.meshgrid(x_obj, y_obj, [ball_size])).T.reshape(-1, 3))
      actions.append(np.array(np.meshgrid(x_goal, y_goal, [ball_size])).T.reshape(-1, 3))
    
    actions = np.concatenate(actions)

    for action in actions:
      x, y, r = ImgToObj.phyreActionToPixelAction(action)
      action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, 0), (x-r, 0)])
      
      if (goal_center[0] - object_center[0]) * (object_center[0] - x) > 0 and rect_intersect(object_bb, action_bb):
        if goal_center[0] < object_center[0] and object_center[0] > x:
          print("Wrong")
          assert(1==0)
        elif goal_center[0] > object_center[0] and object_center[0] < x:
          print("Wrong")
          assert(1==0)
        action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, object_center[1]), (x-r, object_center[1])])
        tl = action_bb[0, :].astype(int)
        br = action_bb[2, :].astype(int)
        rect_img = initial_scene[br[1]:tl[1],tl[0]:br[0]]
        if ImgToObj.Layer.static_body not in rect_img:
          sim_result = simulator.simulate_action(task_index, action, need_images=False)
          if not sim_result.status.is_invalid():
            good_action_count += 1
          if(sim_result.status.is_solved()):
            solved_action_count += 1
      elif goal_type == ImgToObj.Layer.dynamic_goal.value:
        if (object_center[0] - goal_center[0]) * (goal_center[0] - x) > 0 and ImgToObj.rect_intersect(goal_bb, action_bb):
          action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, goal_center[1]), (x-r, goal_center[1])])
          tl = action_bb[0, :].astype(int)
          br = action_bb[2, :].astype(int)
          rect_img = initial_scene[br[1]:tl[1],tl[0]:br[0]]
          if ImgToObj.Layer.static_body not in rect_img:
            sim_result = simulator.simulate_action(task_index, action, need_images=False)
            if not sim_result.status.is_invalid():
              good_action_count += 1
            if(sim_result.status.is_solved()):
              solved_action_count += 1

    results.append({'num_good': good_action_count,
                    'num_solved': solved_action_count, 'num_total': actions.shape[0]})

  return results


tier = 'ball'
cache = phyre.get_default_100k_cache(tier)
task_ids = list(cache.task_ids)
task_ids.sort()
task_ids = task_ids
simulator = phyre.initialize_simulator(task_ids, tier)
initial_scenes = simulator.initial_scenes

print(len(task_ids))

pool_count = 8
pool = multiprocessing.Pool(pool_count)
partial_worker = partial(
    count_good_actions,
    tier=tier)

t0 = time.time()
results_list = pool.imap(partial_worker, chunkify(task_ids, pool_count))
total_actions = []
reduction_count = []
percent_solved = []
no_good_count = 0
no_sol_count = 0

for results in results_list:
  for result in results:
    total_actions.append(result['num_total'])
    if result['num_good'] != 0:
      reduction_count.append(result['num_good'])
      percent_solved.append(result['num_solved'] / result['num_good'])
    else:
      no_good_count += 1
    if result['num_solved'] == 0:
      no_sol_count += 1
t1 = time.time()
print((t1-t0)/len(task_ids), "Avg Time")

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")

f = open("./stats/simple_agent_stats{}.txt".format(dt_string), "w+")

print("Total Actions Mean:", np.mean(total_actions), file=f)
print("Reduction Mean:", np.mean(reduction_count),
      "STD:", np.std(reduction_count), file=f)
print("Reduction Max:", np.max(reduction_count), "Reduction Min:", np.min(reduction_count), file=f)
print("Percent Solved Mean:", np.mean(percent_solved),
      "STD:", np.std(percent_solved), file=f)
print("Percent With No Good action:", no_good_count/len(task_ids), file=f)
print("Percent With No solution:", no_sol_count/len(task_ids), file=f)
