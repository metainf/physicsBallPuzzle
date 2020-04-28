import random
import faulthandler
import multiprocessing
from functools import partial
from collections import Counter
from datetime import datetime

import numpy as np

from tqdm import tqdm

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

import phyre

import ImgToObj


def evalAction(args,initial_img, seq_data, goal_type, simulator, task_index, evaluator):
  x = args['x']
  y = args['y']
  r = args['r']
  stride = 5
  goal = 3 * 60/stride
  score = 1
  solved = False
  if ImgToObj.check_seq_action_intersect(initial_img,seq_data, 100, goal_type,np.array([x,y,r])):
    sim_result = simulator.simulate_action(
        task_index, [x, y, r], need_images=True, stride=stride)
    evaluator.maybe_log_attempt(task_index, sim_result.status)
    if not sim_result.status.is_invalid():
      seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
      score = 1.0 - np.linalg.norm(seq_data['object'][-1]['centroid'] - seq_data['goal'][-1]['centroid']) / 256.0
      score += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
      score = -score
    solved = sim_result.status.is_solved()
  return{'loss': score, 'status': STATUS_OK, 'solved': solved}


def evaluate_agent(tasks, tier):
  """Evaluates the random agent on the given tasks/tier.

  Args:
      tasks: A list of task instances (strings) in the split to evaluate.
      tier: A string of the action tier.

  Returns:
      A Evaluator object updated with the results of all the siulations.
  """

  # Create a simulator for the task and tier.
  simulator = phyre.initialize_simulator(tasks, tier)
  evaluator = phyre.Evaluator(tasks)
  task_data_dict = phyre.loader.load_compiled_task_dict()
  empty_action = phyre.simulator.scene_if.UserInput()
  tasks_solved = 0
  for task_index in tqdm(range(len(tasks)), desc='Evaluate tasks'):
    task_id = tasks[task_index]
    task_data = task_data_dict[task_id]
    _, _, images, _ = phyre.simulator.magic_ponies(
        task_data, empty_action, need_images=True, stride=100)

    evaluator.maybe_log_attempt(task_index, phyre.simulation_cache.NOT_SOLVED)

    seq_data = ImgToObj.getObjectAndGoalSequence(images)
    goal_type = ImgToObj.Layer.dynamic_goal.value
    if goal_type not in images[0]:
      goal_type = ImgToObj.Layer.static_goal.value
    
    simFunc = partial(evalAction,initial_img=images[0], seq_data=seq_data, goal_type=goal_type, simulator=simulator,
                      task_index=task_index, evaluator=evaluator)
    space = {
        'x': hp.uniform('x', 0, 1),
        'y': hp.uniform('y', 0, 1),
        'r': hp.uniform('r', 0, 1),
    }
    trials = Trials()

    max_evals = 0

    solved_task = False
    best_score = 0

    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS and not solved_task:
      max_evals += phyre.MAX_TEST_ATTEMPTS - evaluator.get_attempts_for_task(task_index)
      if best_score > -1.0:
        best = fmin(simFunc,
                space=space,
                algo=hyperopt.rand.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate= random.seed(0),
                show_progressbar=False)
      else:
        best = fmin(simFunc,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate= random.seed(0),
                show_progressbar=False)
      counter = Counter(result['solved'] for result in trials.results)
      solved_task = counter[True] > 0
      tasks_solved += solved_task
      best_score = trials.best_trial['result']['loss']

  print(tasks_solved, "Tasks solved out of ", len(tasks), "Total Tasks")
  return (evaluator.get_aucess(), tasks_solved,len(tasks))

def worker(fold_id,eval_setup):
    __, __, test_tasks = phyre.get_fold(eval_setup, fold_id)
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    tasks = test_tasks
    return evaluate_agent(tasks, action_tier)

faulthandler.enable()
random.seed(0)
eval_setups = ['ball_cross_template', 'ball_within_template']
#fold_ids = list(range(0, 10))
#eval_setups = ['ball_cross_template']
fold_ids = list(range(0, 10))
print('eval setups', eval_setups)
print('fold ids', fold_ids)
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")

f = open("simple_hyperopt_seq_agent_results{}.csv".format(dt_string), "w+")
print('eval_setup,fold_id,AUCESS,Solved,Total,Percent_Solved', file=f)

for eval_setup in eval_setups:
  pool = multiprocessing.Pool(6)
  partial_worker = partial(
    worker,
    eval_setup=eval_setup)
  results = pool.imap(partial_worker, fold_ids)
  for fold_id,result in enumerate(results):
    print('{},{},{},{},{},{}'.format(eval_setup, fold_id, result[0],result[1],result[2],float(result[1])/result[2]), file=f)
