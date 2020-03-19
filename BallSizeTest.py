from functools import partial
import multiprocessing
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np

import phyre

def chunkify(lst, n):
  return [lst[i::n] for i in range(n)]

def count_ball_sizes(task_ids, tier,ball_sizes,num_pos):
  cache = phyre.get_default_100k_cache(tier)
  simulator = phyre.initialize_simulator(task_ids, tier)
  num_solved = 0
  positions = np.linspace(0,1,num_pos)
  for task_index, task_id in tqdm(enumerate(task_ids),desc='Evaluate Tasks',total = len(task_ids)):
    statuses = cache.load_simulation_states(task_id)
    solved_actions = cache.action_array[statuses==phyre.simulation_cache.SOLVED,:]
    solved_actions[:,2] = ball_sizes[abs(solved_actions[:,2][None, :] - ball_sizes[:, None]).argmin(axis=0)]
    for solved_action in solved_actions:
      sim_result = simulator.simulate_action(
      task_index, solved_action, need_images=False)
      if sim_result.status.is_solved():
        num_solved += 1
        break
  return num_solved

tier = 'ball'
cache = phyre.get_default_100k_cache(tier)
task_ids = list(cache.task_ids)

ball_sizes = np.linspace(0.1,1,5)
num_pos = 25

pool_count = 8
pool = multiprocessing.Pool(pool_count)
partial_worker = partial(
    count_ball_sizes,
    tier=tier,
    ball_sizes=ball_sizes,
    num_pos = num_pos)

t0 = time.time()
results_list = pool.imap(partial_worker, chunkify(task_ids, pool_count))

total_num_solved = 0
for results in results_list:
  total_num_solved += results

t1 = time.time()
print((t1-t0)/len(task_ids), "Avg Time")

now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H%M%S")
f = open("./stats/ball_sizes_stats{}.txt".format(dt_string), "w+")
print("Percent Solved:",total_num_solved/len(task_ids), file=f)
print("Ball Sizes:",ball_sizes,file=f)
print("XY spacing:",num_pos,file=f)