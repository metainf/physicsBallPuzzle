import time

import numpy as np
from numpy.polynomial import polynomial as P

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import phyre

import ImgToObj

from shapely.geometry import Polygon
from shapely.ops import unary_union

eval_setup = 'ball_cross_template'
tier = 'ball'
task_str = '00004:243'
stride = 5

task_dict = phyre.loader.load_compiled_task_dict()

t0 = time.time()
task = task_dict[task_str]
t1 = time.time()
print(t1-t0,"Load Time")

t0 = time.time()
_, _, images,_ = phyre.simulator.magic_ponies(task, phyre.simulator.scene_if.UserInput(),need_images=True,stride = stride)
t1 = time.time()
print(t1-t0,"Sim Time")

t0 = time.time()
seq_data = ImgToObj.getObjectAndGoalSequence(images)
t1 = time.time()
print(t1-t0,"Sequence Contour Finding Time")

t0 = time.time()
object_bbs = []
for object_data in seq_data['goal']:
  object_bbs.append(Polygon(object_data['data']))
object_shape = unary_union(object_bbs)
t1 = time.time()
print(t1-t0,"Sequence Merge Finding Time")

x,y = object_shape.exterior.xy
plt.plot(x,y)
plt.show()
