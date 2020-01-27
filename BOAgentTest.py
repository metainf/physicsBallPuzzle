import math
import random
import time
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import phyre

import ImgToObj

from bayes_opt import BayesianOptimization, UtilityFunction



def evalAction(input, simulator):
  x = input['x']
  y = input['y']
  r = input['r']
  stride = 5
  goal = 3 * 60.0/stride
  sim_result = simulator.simulate_action(
      0, [x, y, r], need_images=True, stride=stride)
  score = 0
  if not sim_result.status.is_invalid():
    score = ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
  return {'score': score, 'solved': sim_result.status.is_solved(), 'valid': not sim_result.status.is_invalid()}


eval_setup = 'ball_cross_template'
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

task_str = '00000:000'

simulator = phyre.initialize_simulator([task_str], action_tier)

pbounds = {'x': (0, 1), 'y': (0, 1),'r':(.1,1)}

optimizer = BayesianOptimization(
    f=None,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

max_evals = 100
curr_evals = 0
found_sol = False

while curr_evals <= max_evals:
  next_point = optimizer.suggest(utility)
  eval_result = evalAction(next_point,simulator)
  print(eval_result)
  optimizer.register(
    params=next_point,
    target=eval_result['score'],
  )
  if eval_result['valid']:
    curr_evals += 1
  if eval_result['solved']:
    found_sol = True
    
  


