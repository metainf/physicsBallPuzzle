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


def count_good_actions(task_ids, tier):
  task_data_dict = phyre.loader.load_compiled_task_dict()
  simulator = phyre.initialize_simulator(task_ids, tier)
  results = []
  stride = 100
  empty_action = phyre.simulator.scene_if.UserInput()
  max_actions = 100
  for task_index in tqdm(range(len(task_ids)), desc='Evaluate tasks'):
    task_id = task_ids[task_index]
    task_data = task_data_dict[task_id]
    statuses = cache.load_simulation_states(task_id)
    _, _, images, _ = phyre.simulator.magic_ponies(
        task_data, empty_action, need_images=True, stride=stride)

    seq_data = ImgToObj.getObjectAndGoalSequence(images)

    discrete_actions = cache.action_array.tolist()
    good_action_count = 0
    solved_action_count = 0

    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in images[0]:
      goal_type = ImgToObj.Layer.static_goal.value

    tested_actions_count = 0
    tested_actions = np.array([[-1,-1,-1]])
    while tested_actions_count < max_actions and solved_action_count <= 0:
      random_action = np.random.random_sample((1,3))
      closest_cache_index = np.argmin(np.linalg.norm(cache.action_array - random_action,axis=1))
      test_action = cache.action_array[closest_cache_index,:]

      test_action_dist = np.linalg.norm(tested_actions - test_action,axis=1)

      if np.any(test_action_dist <= .2) and np.random.random_sample() >= .25:
        continue
      x, y, r = ImgToObj.phyreActionToPixelAction(test_action)
      found_intersect = False
      for frame_index, object_data, goal_data in zip(range(len(seq_data['object'])), seq_data['object'], seq_data['goal']):
        if found_intersect:
          break
        goal_bb = goal_data['bb']
        object_bb = object_data['bb']
        time = frame_index * stride / ImgToObj.FRAME_PER_SEC
        y_time = max(r,y + 1.0/2.0 * ImgToObj.GRAV_PIX_PER_SEC * time * time)
        test_action_bb = [(x-r, y+r), (x+r, y+r), (x+r, y-r), (x-r, y-r)]
        test_action_time_bb = [(x-r, y_time+r), (x+r, y_time+r), (x+r, y_time-r), (x-r, y_time-r)]
        if rect_intersect(object_bb, test_action_bb) or rect_intersect(object_bb, test_action_time_bb):
          if statuses[closest_cache_index] != phyre.simulation_cache.INVALID:
            good_action_count += 1
            tested_actions_count += 1
            tested_actions = np.concatenate((tested_actions,test_action[np.newaxis,:]),0)
            found_intersect = True
          if statuses[closest_cache_index] == phyre.simulation_cache.SOLVED:
            solved_action_count += 1
        elif goal_type == ImgToObj.Layer.dynamic_goal.value and (ImgToObj.rect_intersect(goal_bb, test_action_bb) or ImgToObj.rect_intersect(goal_bb, test_action_time_bb)):
          if statuses[closest_cache_index] != phyre.simulation_cache.INVALID:
            good_action_count += 1
            tested_actions_count += 1
            found_intersect = True
            tested_actions = np.concatenate((tested_actions,test_action[np.newaxis,:]),0)
          if statuses[closest_cache_index] == phyre.simulation_cache.SOLVED:
            solved_action_count += 1
    results.append({'num_good': good_action_count,
                    'num_solved': solved_action_count, 'num_total': len(discrete_actions)})

  return results


tier = 'ball'
cache = phyre.get_default_100k_cache(tier)
task_ids = list(cache.task_ids)
task_ids.sort()
task_ids = task_ids
simulator = phyre.initialize_simulator(task_ids, tier)

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

f = open("./stats/random_agent_with_seq_stats{}.txt".format(dt_string), "w+")

print("Total Actions Mean:", np.mean(total_actions), file=f)
print("Reduction Mean:", np.mean(reduction_count),
      "STD:", np.std(reduction_count), file=f)
print("Reduction Max:", np.max(reduction_count), "Reduction Min:", np.min(reduction_count), file=f)
print("Percent Solved Mean:", np.mean(percent_solved),
      "STD:", np.std(percent_solved), file=f)
print("Percent With No Good action:", no_good_count/len(task_ids), file=f)
print("Percent With No solution:", no_sol_count/len(task_ids), file=f)
