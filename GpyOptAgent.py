import random
import faulthandler
import multiprocessing
from functools import partial

import numpy as np

from tqdm import tqdm

import GPyOpt

import phyre

import ImgToObj


def evalAction(args, simulator, task_index, evaluator):
  args = np.squeeze(args)
  x = args[0]
  y = args[1]
  r = args[2]
  stride = 5
  goal = 3 * 60/stride
  sim_result = simulator.simulate_action(
      task_index, [x, y, r], need_images=True, stride=stride)
  evaluator.maybe_log_attempt(task_index, sim_result.status)
  score = 0
  if not sim_result.status.is_invalid():
    score = ImgToObj.objectTouchGoalSequence(sim_result.images)
  if score < goal:
    score = -score + (score - goal) * (score - goal)
  else:
    score = -score
  return {'score': score, 'solved': sim_result.status.is_solved(), 'valid': not sim_result.status.is_invalid()}


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
  assert tuple(tasks) == simulator.task_ids
  tasks_solved = 0
  for task_index in tqdm(range(len(tasks)), desc='Evaluate tasks'):
    domain = [{'name': 'var1', 'type': 'continuous', 'domain': (0, 1)},
              {'name': 'var2', 'type': 'continuous', 'domain': (0, 1)},
              {'name': 'var3', 'type': 'continuous', 'domain': (.1, 1)}]

    X_init = np.array([[0.5, .5, .5]])
    eval_result = evalAction(X_init, simulator, task_index, evaluator)
    Y_init = np.array([[eval_result['score']]])

    X_step = X_init
    Y_step = Y_init

    solved_task = eval_result['solved']
    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS and not solved_task:
      bo_step = GPyOpt.methods.BayesianOptimization(
          f=None, domain=domain, X=X_step, Y=Y_step, de_duplication=True, acquisition_type='MPI', model_type='sparseGP')
      x_next = bo_step.suggest_next_locations()
      eval_result = evalAction(x_next, simulator, task_index, evaluator)
      X_step = np.vstack((X_step, x_next))
      Y_step = np.vstack((Y_step, eval_result['score']))
      #if eval_result['valid']:
      #  print(tasks[task_index],evaluator.get_attempts_for_task(task_index),x_next,eval_result)
      if eval_result['solved']:
        solved_task = True

  print(tasks_solved, "Tasks solved out of ", len(tasks), "Total Tasks")
  return evaluator


def worker(fold_id, eval_setup):
  __, __, test_tasks = phyre.get_fold(eval_setup, fold_id)
  action_tier = phyre.eval_setup_to_action_tier(eval_setup)
  tasks = test_tasks
  evaluator = evaluate_simple_agent(tasks, action_tier)
  return evaluator.get_aucess()


faulthandler.enable()
random.seed(42)
eval_setups = ['ball_cross_template', 'ball_within_template']
# For simplicity, we will just use one fold for evaluation.
fold_ids = list(range(0, 10))
print('eval setups', eval_setups)
print('fold ids', fold_ids)

f = open("simple_GpyOpt_agent_results.csv", "w+")
print('eval_setup,fold_id,AUCESS', file=f)

for eval_setup in eval_setups:
  pool = multiprocessing.Pool(4)
  partial_worker = partial(
      worker,
      eval_setup=eval_setup)
  results = pool.imap(partial_worker, fold_ids)
  for fold_id, result in enumerate(results):
    print('{},{},{}'.format(eval_setup, fold_id, result), file=f)
