import random

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import skewnorm
from scipy.stats import gaussian_kde

import phyre

import ImgToObj

tier = 'ball'
eval_setup = 'ball_cross_template'
fold_id = 2
random.seed(0)

train, dev, test = phyre.get_fold(eval_setup, fold_id)
cache = phyre.get_default_100k_cache(tier)

all_solved_sizes = {}
for task_id in train:
  task_type = task_id.split(":")[0]
  statuses = cache.load_simulation_states(task_id)
  solved_actions = cache.action_array[statuses==phyre.simulation_cache.SOLVED,:]
  solved_sizes = solved_actions[:,2]
  if task_type not in all_solved_sizes:
    all_solved_sizes[task_type] = solved_sizes
  else:
    all_solved_sizes[task_type] = np.concatenate((all_solved_sizes[task_type],solved_sizes),0)

for task_type in all_solved_sizes.keys():
  plt.figure()
  n, bins, patches = plt.hist(all_solved_sizes[task_type], 50, density=True, facecolor='blue', alpha=0.5)
  X = np.linspace(min(all_solved_sizes[task_type]), max(all_solved_sizes[task_type]))
  plt.plot(X, skewnorm.pdf(X, *skewnorm.fit(all_solved_sizes[task_type])))
  kernel = gaussian_kde(all_solved_sizes[task_type],bw_method="silverman")
  plt.plot(X, kernel.pdf(X))
  plt.title(task_type)

plt.show()
