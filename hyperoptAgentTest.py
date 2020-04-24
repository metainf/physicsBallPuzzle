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
    seq_data = ImgToObj.getObjectAndGoalSequence(sim_result.images)
    score = 1.0 - np.linalg.norm(seq_data['object'][-1]['centroid'] - seq_data['goal'][-1]['centroid']) / 256.0
    score += ImgToObj.objectTouchGoalSequence(sim_result.images) / goal
    score = -score
  print(score , sim_result.status.is_solved())
  return{'loss': score, 'status': STATUS_OK, 'solved': sim_result.status.is_solved(), 'valid': not sim_result.status.is_invalid()}


eval_setup = 'ball_cross_template'
action_tier = phyre.eval_setup_to_action_tier(eval_setup)

task_str = '00004:243'

simulator = phyre.initialize_simulator([task_str], action_tier)

simFunc = partial(evalAction, simulator=simulator)
space = {
    'x': hp.uniform('x', 0, 1),
    'y': hp.uniform('y', 0, 1),
    'r': hp.uniform('r', 0, 1),
}

trials = Trials()

max_evals = 0
batch_size = 10
max_trials = 100
found_sol = False
curr_trials = 0
best_score = 0

while curr_trials < max_trials and not found_sol:
  max_evals += batch_size
  if best_score > -1.0:
    best = fmin(simFunc,
                space=space,
                algo=hyperopt.rand.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=random.seed(0))
  else:
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
  best_score = trials.best_trial['result']['loss']


print(curr_trials)
print(trials.best_trial)

