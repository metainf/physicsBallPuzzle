from operator import itemgetter

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
  evaluator = phyre.Evaluator(tasks)
  assert tuple(tasks) == simulator.task_ids
  tasks_solved = 0
  for task_index in tqdm(range(len(tasks)), desc='Evaluate tasks'):
    # Get the initial scene and process it
    initial_scene = simulator.initial_scenes[task_index]
    scene_objects = ImgToObj.phyreToObj(initial_scene)

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

    discrete_actions = simulator.build_discrete_action_space(10000)
    valid_actions = []
    for test_action in discrete_actions:
      x,y,r = ImgToObj.phyreActionToPixelAction(test_action)
      if (goal_centroid[0] - object_centroid[0]) * (object_centroid[0] - x) > 0:
        test_action_bb = [(x-r,y+r),(x+r,y+r),(x+r,0),(x-r,0)]
        if rect_intersect(object_bb,test_action_bb):
          valid_actions.append(test_action)
          """
          fig, axs = plt.subplots()
          axs.imshow(np.full((256,256,3),255))
          patches = []
          for layer_objects in scene_objects:
            for circle in layer_objects['circles']:
              circle1=plt.Circle((circle[0],256.0-circle[1]),radius=circle[2],color='b',fill=True)
              axs.add_artist(circle1)

            circle1=plt.Circle((x,256.0-y),radius=r,color='r',fill=True)
            axs.add_artist(circle1)
            for polygon in layer_objects['polygons']:
              verts = polygon[0]
              verts[:,1] = 256.0-verts[:,1]
              polygon = Polygon(verts,color='b')
              patches.append(polygon)

          verts = np.array(object_bb)
          verts[:,1] = 256.0-verts[:,1]
          polygon = Polygon(verts,color='g',alpha=.5)
          patches.append(polygon)

          verts = np.array(test_action_bb)
          verts[:,1] = 256.0-verts[:,1]
          polygon = Polygon(verts,color='r',alpha=.5)
          patches.append(polygon)

          p1 = PatchCollection(patches)
          axs.add_collection(p1)
          plt.show()
          """
    #print('Number of discrete actions',len(discrete_actions),',','Number of valid actions',len(valid_actions))
    valid_actions.sort(reverse=True,key=itemgetter(2))
    action_index = 0
    solved_task = False
    while evaluator.get_attempts_for_task(task_index) < phyre.MAX_TEST_ATTEMPTS and not solved_task:

      action = valid_actions[action_index%len(valid_actions)]
      action_index += 1
      # Simulate the given action and add the status from taking the action to the evaluator.
      status, _ = simulator.simulate_single(task_index,
                                            action,
                                            need_images=False)
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


eval_setup = 'ball_cross_template'
fold_id = 0  # For simplicity, we will just use one fold for evaluation.
train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
print('Size of resulting splits:\n train:', len(train_tasks), '\n dev:',
      len(dev_tasks), '\n test:', len(test_tasks))

action_tier = phyre.eval_setup_to_action_tier(eval_setup)
print('Action tier for', eval_setup, 'is', action_tier)

tasks = dev_tasks

evaluator = evaluate_random_agent(tasks, action_tier)
print('AUCESS after 100 attempts of random agent on',len(tasks),'tasks of dev set',
      evaluator.get_aucess())

evaluator = evaluate_simple_agent(tasks, action_tier)
print('AUCESS after 100 attempts of simple agent on',len(tasks),'tasks of dev set',
      evaluator.get_aucess())