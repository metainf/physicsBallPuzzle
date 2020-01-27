import math
import random
import time
from functools import partial
from collections import Counter

import numpy as np
from numpy.random import seed

import matplotlib.pyplot as plt

import phyre

import ImgToObj

import GPyOpt

def evalAction(input, simulator):
  input = np.squeeze(input)
  x = input[0]
  y = input[1]
  r = input[2]
  stride = 5
  goal = 3 * 60.0/stride
  sim_result = simulator.simulate_action(
      0, [x, y, r], need_images=True, stride=stride)
  score = 0
  if not sim_result.status.is_invalid():
    score = ImgToObj.objectTouchGoalSequence(sim_result.images)
  if score < goal:
    score = -score + (score - goal) * (score - goal)
  else:
    score = -score
  return {'score': score, 'solved': sim_result.status.is_solved(), 'valid': not sim_result.status.is_invalid()}

seed(42)

eval_setup = 'ball_cross_template'
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

task_str = '00000:000'

simulator = phyre.initialize_simulator([task_str], action_tier)

domain =[{'name': 'var1', 'type': 'continuous', 'domain': (0,1)},
        {'name': 'var2', 'type': 'continuous', 'domain': (0,1)},
        {'name': 'var3', 'type': 'continuous', 'domain': (.1,1)}]

X_init = np.array([[0.5,.5,.5]])
Y_init = np.array([[evalAction(X_init,simulator)['score']]])

X_step = X_init
Y_step = Y_init

max_evals = 100
curr_evals = 0
found_sol = False

while curr_evals <= max_evals and not found_sol:
  bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain, X=X_step, Y=Y_step,de_duplication=True,acquisition_type='MPI')
  x_next = bo_step.suggest_next_locations()
  eval_result = evalAction(x_next,simulator)
  X_step = np.vstack((X_step, x_next))
  Y_step = np.vstack((Y_step, eval_result['score']))
  if eval_result['valid']:
    curr_evals += 1
    print(x_next,eval_result)
  if eval_result['solved']:
    found_sol = True

print(curr_evals)
print(found_sol)
    
  


