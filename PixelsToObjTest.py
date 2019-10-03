import math
import random

import numpy as np
import matplotlib.pyplot as plt

import phyre

eval_setup = 'ball_cross_template'
fold_id = 0  # For simplicity, we will just use one fold for evaluation.
train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
tasks = dev_tasks[:50]
simulator = phyre.initialize_simulator(tasks, action_tier)
task_index = 0  # Note, this is a integer index of task within simulator.task_ids.
task_id = simulator.task_ids[task_index]
initial_scene = simulator.initial_scenes[task_index]
actions = simulator.build_discrete_action_space(max_actions=100)
#action = random.choice(actions)
action = [.5,.5,0.0625]
status, images = simulator.simulate_single(task_index, action, need_images=True)

print('Result of taking action', action, 'on task', tasks[task_index], 'is:',
      status)
print('Does', action, 'solve task', tasks[task_index], '?', status.is_solved())
print('Is', action, 'an invalid action on task', tasks[task_index], '?',
      status.is_invalid())

img_start = phyre.vis.observations_to_float_rgb(images[0])
img_end = phyre.vis.observations_to_float_rgb(images[-1])
fig, axs = plt.subplots(1, 2, figsize=(7, 7))
fig.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)
axs[0].imshow(img_start)
axs[0].title.set_text(f'Start state\nAction solves task: {status.is_solved()}')
axs[0].get_xaxis().set_ticks([])
axs[0].get_yaxis().set_ticks([])

axs[1].imshow(img_end)
axs[1].title.set_text(f'End state\nAction solves task: {status.is_solved()}')
axs[1].get_xaxis().set_ticks([])
axs[1].get_yaxis().set_ticks([]);