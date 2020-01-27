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

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials


def evalAction(input, simulator):
  x = input['x']
  y = input['y']
  r = input['r']
  stride = 5
  goal = 3 * 60/stride
  sim_result = simulator.simulate_action(
      0, [x, y, r], need_images=True, stride=stride)
  score = 0
  if not sim_result.status.is_invalid():
    score = ImgToObj.objectTouchGoalSequence(sim_result.images)
  if score < goal:
    score = -score + (score - goal) * (score - goal)
  else:
    score = -score
  return{'loss': score, 'status': STATUS_OK, 'solved': sim_result.status.is_solved(), 'valid': not sim_result.status.is_invalid()}


eval_setup = 'ball_cross_template'
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

task_str = '00000:000'

simulator = phyre.initialize_simulator([task_str], action_tier)

simFunc = partial(evalAction, simulator=simulator)
space = {
    'x': hp.uniform('x', 0, 1),
    'y': hp.uniform('y', 0, 1),
    'r': 1-hp.loguniform('r', np.log(0.1), np.log(1))
}

trials = Trials()

max_evals = 0
batch_size = 10
max_trials = 100
found_sol = False
curr_trials = 0

while curr_trials < max_trials and not found_sol:
  max_evals += batch_size
  best = fmin(simFunc,
              space=space,
              algo=tpe.suggest,
              max_evals=max_evals,
              trials=trials,
              rstate=random.seed(0))
  counter = Counter(result['valid'] for result in trials.results)
  curr_trials = counter[True]
  counter = Counter(result['solved'] for result in trials.results)
  found_sol = counter[True] > 0


print(curr_trials)
print(trials.best_trial)

