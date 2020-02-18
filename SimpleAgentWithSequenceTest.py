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
  ball_sizes = np.linspace(0.25, 1, 3)

  results = []
  stride = 100
  empty_action = phyre.simulator.scene_if.UserInput()
  
  for task_index,task_id in tqdm(enumerate(task_ids),total = len(task_ids), desc='Evaluate tasks'):
    task_data = task_data_dict[task_id]
    _, _, images, _ = phyre.simulator.magic_ponies(
        task_data, empty_action, need_images=True, stride=stride)

    seq_data = ImgToObj.getObjectAndGoalSequence(images)

    good_action_count = 0
    solved_action_count = 0

    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in images[0]:
      goal_type = ImgToObj.Layer.static_goal.value

    actions = []

    for ball_size in ball_sizes:
      for frame_index, object_data, goal_data in zip(range(len(seq_data['object'])), seq_data['object'], seq_data['goal']):
        time = frame_index * stride / ImgToObj.FRAME_PER_SEC
        goal_bb = goal_data['bb']
        goal_center = goal_data['centroid']
        object_bb = object_data['bb']
        object_center = object_data['centroid']
        r = ImgToObj.phyreRadiusToPixelRadius(ball_size)
        x_spacing = 6
        y_spacing = 6
        
        left_bound_obj = max(r,min(object_bb[:,0]) - r/2)
        right_bound_obj = min(256-r,max(object_bb[:,0]) + r/2)
        x_obj = np.linspace(left_bound_obj/256,right_bound_obj/256,x_spacing)
        bottom_bound_obj = max(r,min(object_bb[:,1]) - r/2)
        top_bound_obj = min(256-r,max(object_bb[:,1]) + r/2)
        y_obj = np.linspace(bottom_bound_obj/256,top_bound_obj/256,y_spacing)

        bottom_bound_obj_time = max(r,bottom_bound_obj - 1.0/2.0 * ImgToObj.GRAV_PIX_PER_SEC * time * time)
        top_bound_obj_time = min(256-r,top_bound_obj - 1.0/2.0 * ImgToObj.GRAV_PIX_PER_SEC * time * time)
        y_obj_time = np.linspace(bottom_bound_obj_time/256,top_bound_obj_time/256,y_spacing)

        actions.append(np.array(np.meshgrid(x_obj, y_obj, [ball_size])).T.reshape(-1, 3))
        actions.append(np.array(np.meshgrid(x_obj, y_obj_time, [ball_size])).T.reshape(-1, 3))

        if goal_type == ImgToObj.Layer.dynamic_goal.value:
          left_bound_goal = max(r,min(goal_bb[:,0]) - r/2)
          right_bound_goal = min(256-r,max(goal_bb[:,0]) + r/2)
          x_goal = np.linspace(left_bound_goal/256,right_bound_goal/256,x_spacing)
          bottom_bound_goal = max(r,min(goal_bb[:,1]) - r/2)
          top_bound_goal = min(256-r,max(goal_bb[:,1]) + r/2)
          y_goal = np.linspace(bottom_bound_goal/256,top_bound_goal/256,y_spacing)

          bottom_bound_goal_time = max(r,bottom_bound_goal - 1.0/2.0 * ImgToObj.GRAV_PIX_PER_SEC * time * time)
          top_bound_goal_time = min(256-r,top_bound_goal - 1.0/2.0 * ImgToObj.GRAV_PIX_PER_SEC * time * time)
          y_obj_time = np.linspace(bottom_bound_goal_time/256,top_bound_goal_time/256,y_spacing)

          actions.append(np.array(np.meshgrid(x_goal, y_goal, [ball_size])).T.reshape(-1, 3))
          actions.append(np.array(np.meshgrid(x_goal, y_obj_time, [ball_size])).T.reshape(-1, 3))
      
    actions = np.concatenate(actions)

    for action in actions:
      sim_result = simulator.simulate_action(task_index, action, need_images=False)
      if not sim_result.status.is_invalid():
        good_action_count += 1
        found_intersect = True
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

print(len(task_ids))

pool_count = 8
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

for results in results_list:
  for result in results:
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

f = open("./stats/simple_agent_with_seq_stats{}.txt".format(dt_string), "w+")

print("Reduction Mean:", np.mean(reduction_count),
      "STD:", np.std(reduction_count), file=f)
print("Reduction Min:", np.min(reduction_count),
      "Reduction Max:", np.max(reduction_count), file=f)
print("Percent Solved Mean:", np.mean(percent_solved),
      "STD:", np.std(percent_solved), file=f)
print("Percent With No Good action:", no_good_count/len(task_ids), file=f)
print("Percent With No solution:", no_sol_count/len(task_ids), file=f)
