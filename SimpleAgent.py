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
  l1 = rect1[0, :]
  r1 = rect1[2, :]
  l2 = rect2[0, :]
  r2 = rect2[2, :]
  # If one rectangle is on left side of other
  if(l1[0] > r2[0] or l2[0] > r1[0]):
    return False

  # If one rectangle is above other
  if(l1[1] < r2[1] or l2[1] < r1[1]):
    return False

  return True


def evaluate_simple_agent(tasks, tier):
  """Evaluates the random agent on the given tasks/tier.

  Args:
      tasks: A list of task instances (strings) in the split to evaluate.
      tier: A string of the action tier.

  Returns:
      A Evaluator object updated with the results of all the siulations.
  """

  # Create a simulator for the task and tier.
  ball_sizes = np.linspace(0.01, 1, 5)
  pos = np.linspace(0, 1, 50)

  actions = np.array(np.meshgrid(pos, pos, ball_sizes)).T.reshape(-1, 3)

  simulator = phyre.initialize_simulator(tasks, tier)
  cache = phyre.get_default_100k_cache(tier)
  evaluator = phyre.Evaluator(tasks)
  assert tuple(tasks) == simulator.task_ids
  tasks_solved = 0

  for task_index in tqdm(range(len(tasks)), desc='Evaluate tasks'):
    # Get the initial scene and process it
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

    solved_task = False

    good_actions = []

    for action in actions:
      x, y, r = ImgToObj.phyreActionToPixelAction(action)
      action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, 0), (x-r, 0)])
      if (goal_center[0] - object_center[0]) * (object_center[0] - x) > 0:
        if rect_intersect(object_bb, action_bb):
          good_actions.append(action)
      elif goal_type == ImgToObj.Layer.dynamic_goal.value:
        if (object_center[0] - goal_center[0]) * (goal_center[0] - x) > 0:
          if ImgToObj.rect_intersect(goal_bb, action_bb):
            good_actions.append(action)


    good_actions.sort(reverse=True, key=lambda x: x[2])

    good_action_index = 0

    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS and not solved_task:
      action = [.5, .5, .5]
      if good_action_index < len(good_actions):
        action = good_actions[good_action_index]
        good_action_index += 1
      else:
        action_index = random.randint(0, cache.action_array.shape[0]-1)
        action = cache.action_array[action_index, :]

      sim_result = simulator.simulate_action(
          task_index, action, need_images=False)
      evaluator.maybe_log_attempt(task_index, sim_result.status)

      if sim_result.status.is_solved():
        solved_task = True
        tasks_solved += 1
  print(tasks_solved, "Tasks solved out of ", len(tasks), "Total Tasks")
  return evaluator


def worker(fold_id, eval_setup):
  __, __, test_tasks = phyre.get_fold(eval_setup, fold_id)
  action_tier = phyre.eval_setup_to_action_tier(eval_setup)
  tasks = test_tasks
  evaluator = evaluate_simple_agent(tasks, action_tier)
  return evaluator.get_aucess()


faulthandler.enable()
random.seed(0)
np.random.seed(0)
eval_setups = ['ball_cross_template', 'ball_within_template']
fold_ids = list(range(0, 10))
print('eval setups', eval_setups)
print('fold ids', fold_ids)

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")

f = open("simple_agent_results{}.csv".format(dt_string), "w+")
print('eval_setup,fold_id,AUCESS', file=f)

for eval_setup in eval_setups:
  pool = multiprocessing.Pool(3)
  partial_worker = partial(
      worker,
      eval_setup=eval_setup)
  results = pool.imap(partial_worker, list(range(0, 10)))
  for fold_id, result in enumerate(results):
    print('{},{},{}'.format(eval_setup, fold_id, result), file=f)
