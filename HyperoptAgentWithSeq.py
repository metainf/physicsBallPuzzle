import random
import faulthandler
import multiprocessing
from functools import partial
from collections import Counter
from datetime import datetime

import numpy as np

from tqdm import tqdm

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials

import phyre

import ImgToObj


def evalAction(args, seq_data, goal_type, simulator, task_index, evaluator):
  x = args['x']
  y = args['y']
  r = args['r']
  stride = 5
  goal = 3 * 60/stride
  score = 0
  score = -score + (score - goal) * (score - goal)
  solved = False
  if ImgToObj.check_seq_action_intersect(seq_data, 100, goal_type,np.array([x,y,r])):
    sim_result = simulator.simulate_action(
        task_index, [x, y, r], need_images=True, stride=stride)
    evaluator.maybe_log_attempt(task_index, sim_result.status)
    if not sim_result.status.is_invalid():
        score = ImgToObj.objectTouchGoalSequence(sim_result.images)
    if score < goal:
        score = -score + (score - goal) * (score - goal)
    else:
        score = -score
    solved = sim_result.status.is_solved()
  return{'loss': score, 'status': STATUS_OK, 'solved': solved}


def evaluate_simple_agent(tasks, tier):
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
    
    simFunc = partial(evalAction, seq_data=seq_data, goal_type=goal_type, simulator=simulator,
                      task_index=task_index, evaluator=evaluator)
    space = {
        'x': hp.uniform('x', 0, 1),
        'y': hp.uniform('y', 0, 1),
        'r': hp.uniform('r', 0, 1),
    }
    trials = Trials()

    max_evals = 0

    solved_task = False
    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS and not solved_task:
      max_evals += phyre.MAX_TEST_ATTEMPTS - evaluator.get_attempts_for_task(task_index)
      best = fmin(simFunc,
              space=space,
              algo=tpe.suggest,
              max_evals=max_evals,
              trials=trials,
              rstate= random.seed(0),
              show_progressbar=False)
      counter = Counter(result['solved'] for result in trials.results)
      solved_task = counter[True] > 0
      if solved_task:
        tasks_solved += 1

  print(tasks_solved, "Tasks solved out of ", len(tasks), "Total Tasks")
  return evaluator

def worker(fold_id,eval_setup):
    __, __, test_tasks = phyre.get_fold(eval_setup, fold_id)
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    tasks = test_tasks
    evaluator = evaluate_simple_agent(tasks, action_tier)
    return evaluator.get_aucess()

faulthandler.enable()
random.seed(0)
eval_setups = ['ball_cross_template', 'ball_within_template']
fold_ids = list(range(0, 10))
print('eval setups', eval_setups)
print('fold ids', fold_ids)
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")

f = open("simple_hyperopt_agent_results{}.csv".format(dt_string), "w+")
print('eval_setup,fold_id,AUCESS', file=f)

for eval_setup in eval_setups:
  pool = multiprocessing.Pool(6)
  partial_worker = partial(
    worker,
    eval_setup=eval_setup)
  results = pool.imap(partial_worker, list(range(0, 10)))
  for fold_id,result in enumerate(results):
    print('{},{},{}'.format(eval_setup, fold_id, result), file=f)
