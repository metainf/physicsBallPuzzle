import random
import faulthandler

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import phyre

import ImgToObj

def circle_bounding_box(circle):
  x = circle[0]
  y = circle[1]
  r = circle[2]
  return [(x-r,y+r),(x+r,y+r),(x+r,y-r),(x-r,y-r)]

def polygon_bounding_box(polygon):
  min_x = np.amin(polygon[:,0])
  max_x = np.amax(polygon[:,0])

  min_y = np.amin(polygon[:,1])
  max_y = np.amax(polygon[:,1])

  return [(min_x,max_y),(max_x,max_y),(max_x,min_y),(min_x,min_y)]

def rect_intersect(rect1,rect2):
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
  cache = phyre.get_default_100k_cache(tier)
  evaluator = phyre.Evaluator(tasks)
  assert tuple(tasks) == simulator.task_ids
  tasks_solved = 0
  for task_index in tqdm(range(len(tasks)), desc='Evaluate tasks'):
    # Get the initial scene and process it
    #print(tasks[task_index])
    initial_scene = simulator.initial_scenes[task_index]
    scene_objects = ImgToObj.phyreToObj(initial_scene)
    #print("Finished parsing scene")
    task_statuses = cache.load_simulation_states(tasks[task_index])


    # Get the centroid of the object
    object_layer_info = scene_objects[ImgToObj.Layer.object.value]
    object_centroid = [0,0]
    object_bb = None
    if len(object_layer_info['circles']) > 0:
      object_centroid[0] = object_layer_info['circles'][0][0]
      object_centroid[1] = object_layer_info['circles'][0][1]
      object_bb = circle_bounding_box(object_layer_info['circles'][0])
    elif len(object_layer_info['polygons']) > 0:
      center = np.mean(object_layer_info['polygons'][0][0].astype(float),axis=0)
      object_centroid[0] = center[0]
      object_centroid[1] = center[1]
      object_bb = polygon_bounding_box(object_layer_info['polygons'][0][0].astype(float))

    # Get the centroid of the goal
    dynamic_goal_layer_info = scene_objects[ImgToObj.Layer.dynamic_goal.value]
    static_goal_layer_info = scene_objects[ImgToObj.Layer.static_goal.value]
    goal_centroid = [0,0]

    if len(dynamic_goal_layer_info['circles']) > 0:
      goal_centroid[0] = dynamic_goal_layer_info['circles'][0][0]
      goal_centroid[1] = dynamic_goal_layer_info['circles'][0][1]
    elif len(dynamic_goal_layer_info['polygons']) > 0:
      center = np.mean(dynamic_goal_layer_info['polygons'][0][0].astype(float),axis=0)
      goal_centroid[0] = center[0]
      goal_centroid[1] = center[1]
    elif len(static_goal_layer_info['circles']) > 0:
      goal_centroid[0] = static_goal_layer_info['circles'][0][0]
      goal_centroid[1] = static_goal_layer_info['circles'][0][1]
    elif len(static_goal_layer_info['polygons']) > 0:
      center = np.mean(static_goal_layer_info['polygons'][0][0].astype(float),axis=0)
      goal_centroid[0] = center[0]
      goal_centroid[1] = center[1]

    #discrete_actions = simulator.build_discrete_action_space(10000)
    discrete_actions = cache.action_array.tolist()
    valid_actions = []
    for action_id,test_action in enumerate(discrete_actions):
      x,y,r = ImgToObj.phyreActionToPixelAction(test_action)
      if (goal_centroid[0] - object_centroid[0]) * (object_centroid[0] - x) > 0 and task_statuses[action_id] != phyre.simulation_cache.INVALID:
        test_action_bb = [(x-r,y+r),(x+r,y+r),(x+r,0),(x-r,0)]
        if rect_intersect(object_bb,test_action_bb):
          valid_actions.append((test_action,action_id))
    #print('Number of discrete actions',len(discrete_actions),',','Number of valid actions',len(valid_actions))
    valid_actions.sort(reverse=True,key=lambda x: x[0][2])
    valid_action_index = 0
    solved_task = False
    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS and not solved_task:

      if len(valid_actions) > 0:
        action_index = valid_actions[valid_action_index%len(valid_actions)][1]
        valid_action_index += 1
      else:
        action_index = random.randint(0, len(cache))
      # Simulate the given action and add the status from taking the action to the evaluator.
      #status, _ = simulator.simulate_single(task_index,
      #                                      action,
      #                                      need_images=False)
      status = phyre.SimulationStatus(task_statuses[action_index])
      #print('Does', action, 'solve task', tasks[task_index], '?', status.is_solved())
      if(status.is_solved()):
        solved_task = True
        tasks_solved += 1
      evaluator.maybe_log_attempt(task_index, status)
  print(tasks_solved,"Tasks solved out of ",len(tasks),"Total Tasks")
  return evaluator

def evaluate_random_agent(tasks, tier):
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
  for task_index in tqdm(range(len(tasks)), desc='Evaluate tasks'):
      while evaluator.get_attempts_for_task(
              task_index) < phyre.MAX_TEST_ATTEMPTS:
          # Sample a random valid action from the simulator for the given action space.
          action = simulator.sample()
          # Simulate the given action and add the status from taking the action to the evaluator.
          status, _ = simulator.simulate_single(task_index,
                                                action,
                                                need_images=False)
          evaluator.maybe_log_attempt(task_index, status)
  return evaluator


faulthandler.enable()
random.seed(0)
eval_setups = ['ball_cross_template', 'ball_within_template']
fold_ids = list(range(0,10))  # For simplicity, we will just use one fold for evaluation.
print('eval setups',eval_setups)
print('fold ids',fold_ids)

f = open("simple_agent_results.csv","w+")
print('eval_setup,fold_id,AUCESS',file=f)

for eval_setup in eval_setups:
  for fold_id in fold_ids:
    print('Eval Setup:',eval_setup,'Fold Id:',fold_id)
    train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
    action_tier = phyre.eval_setup_to_action_tier(eval_setup)
    tasks = test_tasks
    evaluator = evaluate_simple_agent(tasks, action_tier)
    print('AUCESS after 100 attempts of simple agent on',len(tasks),'tasks',
          evaluator.get_aucess())
    print('{},{},{}'.format(eval_setup,fold_id,evaluator.get_aucess()),file=f)