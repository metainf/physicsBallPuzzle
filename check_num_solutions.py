import numpy as np

import phyre
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



eval_setup = 'ball_cross_template'
tier = phyre.eval_setup_to_action_tier(eval_setup)
cache = phyre.get_default_100k_cache(tier)
task = "00013:064"
task_statuses = cache.load_simulation_states(task)
print('Share of SOLVED statuses for task:',task,(task_statuses == phyre.SimulationStatus.SOLVED).mean())

solution_index = np.argwhere(task_statuses == phyre.SimulationStatus.SOLVED)
fig = plt.figure()
ax = plt.axes(projection='3d')

action_array = cache.action_array
print(action_array[solution_index,:])


ax.scatter3D(action_array[solution_index,0], action_array[solution_index,1], action_array[solution_index,2]);
#ax.scatter(action_array[solution_index,0],action_array[solution_index,1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()



