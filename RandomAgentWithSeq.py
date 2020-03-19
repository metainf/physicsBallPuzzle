import random
import faulthandler
from functools import partial
import multiprocessing
from datetime import datetime

import numpy as np

from tqdm import tqdm

import phyre

import ImgToObj


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


def evaluate_agent(task_ids, tier):
  evaluator = phyre.Evaluator(task_ids)
  task_data_dict = phyre.loader.load_compiled_task_dict()
  stride = 100
  empty_action = phyre.simulator.scene_if.UserInput()
  cache = phyre.get_default_100k_cache(tier)
  tasks_solved = 0    

  for task_index in tqdm(range(len(task_ids)), desc='Evaluate tasks'):
    task_id = task_ids[task_index]
    task_data = task_data_dict[task_id]
    statuses = cache.load_simulation_states(task_id)
    _, _, images, _ = phyre.simulator.magic_ponies(
        task_data, empty_action, need_images=True, stride=stride)

    evaluator.maybe_log_attempt(task_index, phyre.simulation_cache.NOT_SOLVED)

    seq_data = ImgToObj.getObjectAndGoalSequence(images)

    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in images[0]:
      goal_type = ImgToObj.Layer.static_goal.value

    tested_actions = np.array([[-1,-1,-1]])

    solved_task = False

    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS and not solved_task:
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
            found_intersect = True
            evaluator.maybe_log_attempt(task_index, statuses[closest_cache_index])
          if statuses[closest_cache_index] == phyre.simulation_cache.SOLVED:
            solved_task = True
            tasks_solved += 1
        elif goal_type == ImgToObj.Layer.dynamic_goal.value and (ImgToObj.rect_intersect(goal_bb, test_action_bb) or ImgToObj.rect_intersect(goal_bb, test_action_time_bb)):
          if statuses[closest_cache_index] != phyre.simulation_cache.INVALID:
            found_intersect = True
            evaluator.maybe_log_attempt(task_index, statuses[closest_cache_index])
          if statuses[closest_cache_index] == phyre.simulation_cache.SOLVED:
            solved_task = True
            tasks_solved += 1

  print(tasks_solved, "Tasks solved out of ", len(task_ids), "Total Tasks")
  return (evaluator.get_aucess(),tasks_solved,len(task_ids))


def worker(fold_id, eval_setup):
  __, __, test_tasks = phyre.get_fold(eval_setup, fold_id)
  action_tier = phyre.eval_setup_to_action_tier(eval_setup)
  tasks = test_tasks
  return evaluate_agent(tasks, action_tier)


faulthandler.enable()
random.seed(0)
np.random.seed(0)
eval_setups = ['ball_cross_template', 'ball_within_template']
fold_ids = list(range(0, 10))
print('eval setups', eval_setups)
print('fold ids', fold_ids)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")

f = open("random_agent_with_seq_results{}.csv".format(dt_string), "w+")
print('eval_setup,fold_id,AUCESS,Solved,Total', file=f)

for eval_setup in eval_setups:
  pool = multiprocessing.Pool(8)
  partial_worker = partial(
      worker,
      eval_setup=eval_setup)
  results = pool.imap(partial_worker, list(range(0, 10)))
  for fold_id, result in enumerate(results):
    print('{},{},{},{},{}'.format(eval_setup, fold_id, result[0],result[1],result[2]), file=f)
