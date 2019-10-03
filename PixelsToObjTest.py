import math
import random

import numpy as np
import matplotlib.pyplot as plt
import pymunk
from pymunk import Vec2d

import phyre

import ImgToObj

eval_setup = 'ball_cross_template'
fold_id = 0  # For simplicity, we will just use one fold for evaluation.
train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
tasks = dev_tasks[:50]
simulator = phyre.initialize_simulator(tasks, action_tier)
task_index = 1  # Note, this is a integer index of task within simulator.task_ids.
task_id = simulator.task_ids[task_index]
initial_scene = simulator.initial_scenes[task_index]
actions = simulator.build_discrete_action_space(max_actions=100)
#action = random.choice(actions)
action = [.5,.5,1]
status, images = simulator.simulate_single(task_index, action, need_images=True)

print('Result of taking action', action, 'on task', tasks[task_index], 'is:',
      status)
print('Does', action, 'solve task', tasks[task_index], '?', status.is_solved())
print('Is', action, 'an invalid action on task', tasks[task_index], '?',
      status.is_invalid())

img_start = phyre.vis.observations_to_float_rgb(images[0])
imgLayers = ImgToObj.phyreImgToBoolLayers(images[0])
fig, axs = plt.subplots(3, 3, figsize=(7, 7))
fig.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)
allCircles = ImgToObj.findCirclesInPhyre(images[0])
for i,ax in enumerate(axs.flatten()):
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
  if i < 7:
    ax.imshow(imgLayers[:,:,i])
    if allCircles[i] is not None:
      circles = allCircles[i]
      for circle in circles[0,:]:
        circle1=plt.Circle((circle[0],circle[1]),radius=circle[2],color='r',fill=False)
        ax.add_artist(circle1)
  if i == 7:
    ax.imshow(img_start)
  elif i == 8:
    ax.imshow(images[0])
plt.show()

# Call this first

# Initialize space
space = pymunk.Space()
space.gravity = (0.0, -900.0)

balls = []

# Make the balls
for circles in allCircles:
  if circles is not None:
    for circle in circles[0,:]:
      mass = 1
      radius = circle[2]
      inertia = pymunk.moment_for_circle(mass, 0, radius, (0,0))
      body = pymunk.Body(mass, inertia)
      body.position = circle[0],circle[1]
      shape = pymunk.Circle(body, radius, Vec2d(0,0))
      shape.elasticity = 0.5
      shape.friction = 0.5
      space.add(body, shape)
      balls.append(shape)

# Make the walls
static_body = space.static_body
static_lines = [pymunk.Segment(static_body, (0.0, 0.0), (0.0, 256.0), 0.0),
                pymunk.Segment(static_body, (0.0, 256.0), (256.0, 256.0), 0.0),
                pymunk.Segment(static_body, (256.0, 256.0), (256.0, 0.0), 0.0),
                pymunk.Segment(static_body, (256.0, 0.0), (0.0, 0.0), 0.0)]
for line in static_lines:
  line.elasticity = 0.5
  line.friction = 0.9
space.add(static_lines)

for i in range(300):
  space.step(1.0 / 60.0)

img_end = phyre.vis.observations_to_float_rgb(images[-1])
fig, ax = plt.subplots()
ax.imshow(img_end)

for ball in balls:
  print(ball.body.position)
  circle1=plt.Circle((ball.body.position.x,256.0-ball.body.position.y),radius=ball.radius,color='b',fill=False)
  ax.add_artist(circle1)

plt.show()
  
