import random
import faulthandler
from functools import partial
import multiprocessing
from datetime import datetime

from scipy.stats import gaussian_kde

import numpy as np

from tqdm import tqdm

import phyre

import ImgToObj

def evaluate_agent(task_ids, tier,ball_size_pdf):
  evaluator = phyre.Evaluator(task_ids)
  simulator = phyre.initialize_simulator(task_ids, tier)
  task_data_dict = phyre.loader.load_compiled_task_dict()
  stride = 100
  empty_action = phyre.simulator.scene_if.UserInput()
  tasks_solved = 0
      
  for task_index in tqdm(range(len(task_ids)), desc='Evaluate tasks'):
    task_id = task_ids[task_index]
    task_type = task_id.split(":")[0]
    task_data = task_data_dict[task_id]
    _, _, images, _ = phyre.simulator.magic_ponies(
        task_data, empty_action, need_images=True, stride=stride)

    evaluator.maybe_log_attempt(task_index, phyre.simulation_cache.NOT_SOLVED)

    seq_data = ImgToObj.getObjectAndGoalSequence(images)

    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in images[0]:
      goal_type = ImgToObj.Layer.static_goal.value

    tested_actions = np.array([[-1,-1,-1,1]])

    solved_task = False

    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS:
      random_action = np.random.random_sample((1,4))
      if task_type in ball_size_pdf:
        random_action[0,2] = ball_size_pdf[task_type].resample(size=1)

      test_action_dist = np.linalg.norm(tested_actions[:,0:3] - random_action[:,0:3],axis=1)

      if np.any(test_action_dist <= tested_actions[:,3]) and np.random.random_sample() >= .25:
        continue
      if ImgToObj.check_seq_action_intersect(seq_data, stride, goal_type,np.squeeze(random_action[0:3])):
        eval_stride = 5
        goal = 3.0 * 60.0/eval_stride
        sim_result = simulator.simulate_action(task_index, np.squeeze(random_action[:,0:3]), need_images=True, stride=eval_stride)
        evaluator.maybe_log_attempt(task_index, sim_result.status)
        if not sim_result.status.is_invalid():
          score = ImgToObj.objectTouchGoalSequence(sim_result.images)
          eval_dist = .1+.3*(score==0)
          random_action[0,3] = eval_dist
          tested_actions = np.concatenate((tested_actions,random_action),0)
          solved_task = sim_result.status.is_solved()
          tasks_solved += solved_task

  print(tasks_solved, "Tasks solved out of ", len(task_ids), "Total Tasks")
  return (evaluator.get_aucess(), tasks_solved,len(task_ids))

def train_ball_sizes(tasks,tier):
  cache = phyre.get_default_100k_cache(tier)
  all_solved_sizes = {}
  for task_id in tasks:
    task_type = task_id.split(":")[0]
    statuses = cache.load_simulation_states(task_id)
    solved_actions = cache.action_array[statuses==phyre.simulation_cache.SOLVED,:]
    solved_sizes = solved_actions[:,2]
    if task_type not in all_solved_sizes:
      all_solved_sizes[task_type] = solved_sizes
    else:
      all_solved_sizes[task_type] = np.concatenate((all_solved_sizes[task_type],solved_sizes),0)
  
  ball_size_pdf = {}
  for task_type in all_solved_sizes.keys():
    ball_size_pdf[task_type] = gaussian_kde(all_solved_sizes[task_type],bw_method="silverman")

  return ball_size_pdf

def worker(fold_id, eval_setup):
  train, dev, test = phyre.get_fold(eval_setup, fold_id)
  action_tier = phyre.eval_setup_to_action_tier(eval_setup)
  ball_size_pdf = train_ball_sizes(train,action_tier)
  return evaluate_agent(test, action_tier,ball_size_pdf)


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
