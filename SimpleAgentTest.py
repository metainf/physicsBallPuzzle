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


def count_good_actions(tasks, tier):
  simulator = phyre.initialize_simulator(tasks, tier)
  results = []
  ball_sizes = np.linspace(0.01, 1, 5)
  pos = np.linspace(0, 1, 50)

  actions = np.array(np.meshgrid(pos, pos, ball_sizes)).T.reshape(-1, 3)
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

    for action in actions:
      x, y, r = ImgToObj.phyreActionToPixelAction(action)
      action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, 0), (x-r, 0)])
      if (goal_center[0] - object_center[0]) * (object_center[0] - x) > 0:
        if rect_intersect(object_bb, action_bb):
          good_action_count += 1
          sim_result = simulator.simulate_action(task_index, action, need_images=False)
          if(sim_result.status.is_solved()):
            solved_action_count += 1
      elif goal_type == ImgToObj.Layer.dynamic_goal.value:
        if (object_center[0] - goal_center[0]) * (goal_center[0] - x) > 0:
          if ImgToObj.rect_intersect(goal_bb, action_bb):
            good_action_count += 1
            sim_result = simulator.simulate_action(task_index, action, need_images=False)
            if(sim_result.status.is_solved()):
              solved_action_count += 1

    results.append({'num_good': good_action_count,
                    'num_solved': solved_action_count, 'num_total': actions.shape[0]})

  return results


tier = 'ball'
cache = phyre.get_default_100k_cache(tier)
task_ids = list(cache.task_ids)
simulator = phyre.initialize_simulator(task_ids, tier)
initial_scenes = simulator.initial_scenes

print(len(task_ids))

pool_count = 4
pool = multiprocessing.Pool(pool_count)
partial_worker = partial(
    count_good_actions,
    tier=tier)

t0 = time.time()
results_list = pool.imap(partial_worker, chunkify(task_ids, pool_count))
reduction_count = []
percent_solved = []
no_good_count = 0
no_sol_count = 0
total_actions = 0

for results in results_list:
  for result in results:
    total_actions = result['num_total'] 
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

f = open("simple_agent_stats{}.txt".format(dt_string), "w+")

print("Total Actions:", total_actions, file=f)
print("Reduction Mean:", np.mean(reduction_count),
      "STD:", np.std(reduction_count), file=f)
print("Percent Solved Mean:", np.mean(percent_solved),
      "STD:", np.std(percent_solved), file=f)
print("Percent With No Good action:", no_good_count/len(task_ids), file=f)
print("Percent With No solution:", no_sol_count/len(task_ids), file=f)
