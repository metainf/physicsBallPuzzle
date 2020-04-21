import matplotlib.pyplot as plt
import numpy as np

import time

import phyre

test_action = np.random.random_sample((1,3))
tier = 'ball'
cache = phyre.get_default_100k_cache(tier)

t0 = time.time()
min_index = np.argmin(np.linalg.norm(cache.action_array - test_action,axis=1))
t1 = time.time()
print((t1-t0), "Search Time")

print(test_action[:,0:2])
print(cache.action_array[min_index,0:2])
print(np.linalg.norm(test_action[:,0:2] - cache.action_array[min_index,0:2]))

print(np.concatenate((cache.action_array[min_index,:][np.newaxis,:],test_action),0))

task_ids = list(cache.task_ids)[0:2]

print(task_ids)

evaluator = phyre.Evaluator(task_ids)
print(evaluator.task_ids)
for i in range(len(task_ids)-1):
    while evaluator.get_attempts_for_task(i) < phyre.MAX_TEST_ATTEMPTS-50:
        evaluator.maybe_log_attempt(i, phyre.simulation_cache.SOLVED)
print(evaluator.get_aucess())

x = np.array([3, 4, 2, 1,5,6])
x[np.argpartition(x, 3)]