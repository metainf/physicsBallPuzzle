import math
import random
import time

import cv2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import pymunk
from pymunk import Vec2d

import phyre

import ImgToObj

eval_setup = 'ball_cross_template'
fold_id = 300  # For simplicity, we will just use one fold for evaluation.
train_tasks, dev_tasks, test_tasks = phyre.get_fold(eval_setup, fold_id)
action_tier = phyre.eval_setup_to_action_tier(eval_setup)
print("Dev Tasks:{}".format(len(dev_tasks)))
tasks = dev_tasks

with open("tasks.txt","w") as f:
  for i,task in enumerate(tasks):
    print(i,task,file=f)

simulator = phyre.initialize_simulator(tasks, action_tier)
task_index = 0  # Note, this is a integer index of task within simulator.task_ids.
task_id = simulator.task_ids[task_index]
initial_scene = simulator.initial_scenes[task_index]
actions = simulator.build_discrete_action_space(max_actions=100)
#action = random.choice(actions)
action = [.7,.75,1.0]
status, images = simulator.simulate_single(task_index, action, need_images=True)

print('Result of taking action', action, 'on task', tasks[task_index], 'is:',
      status)
print('Does', action, 'solve task', tasks[task_index], '?', status.is_solved())
print('Is', action, 'an invalid action on task', tasks[task_index], '?',
      status.is_invalid())

img_start = phyre.vis.observations_to_float_rgb(images[0])

fig, axs = plt.subplots(3, 3, figsize=(7, 7))
fig.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.3)

t0 = time.time()
scene_objects = ImgToObj.phyreToObj(initial_scene,action)
t1 = time.time()
print(t1-t0,"Contour Finding Time")

for i,ax in enumerate(axs.flatten()):
  ax.get_xaxis().set_ticks([])
  ax.get_yaxis().set_ticks([])
  if i < 7:
    colorImg = np.stack((scene_objects[i]['raw_img'],)*3, axis=-1)
    ax.imshow(colorImg)
    for polygon in scene_objects[i]['polygons']:
      ax.scatter(polygon[0][:,0],polygon[0][:,1],s=4,c='r')
      for triangle in polygon[1]['triangles'].tolist():
        ax.plot(polygon[1]['vertices'][triangle,0],polygon[1]['vertices'][triangle,1],'b')
        ax.plot(polygon[1]['vertices'][[triangle[0],triangle[-1]],0],polygon[1]['vertices'][[triangle[0],triangle[-1]],1],'b')
    ax.title.set_text(str(i))
    for circle in scene_objects[i]['circles']:
      print(i,circle)
      circle1=plt.Circle((circle[0],circle[1]),radius=circle[2],color='r',fill=False)
      ax.add_artist(circle1)
  if i == 7:
    ax.imshow(img_start)
  elif i == 8:
    ax.imshow(images[0])
plt.show()

t0 = time.time()
# Initialize space
space = pymunk.Space()
space.gravity = (0.0, -50.0)
space.iterations = 10

COLLISION_OBJECT = 1
COLLISION_GOAL = 2

# Make the walls
static_body = space.static_body
static_lines = [pymunk.Segment(static_body, (0.0, 0.0), (0.0, 256.0), 0.0),
                pymunk.Segment(static_body, (0.0, 256.0), (256.0, 256.0), 0.0),
                pymunk.Segment(static_body, (256.0, 256.0), (256.0, 0.0), 0.0),
                pymunk.Segment(static_body, (256.0, 0.0), (0.0, 0.0), 0.0)]
for line in static_lines:
  line.elasticity = 0.3
  line.friction = 0.6
space.add(static_lines)

# Add the scene objects to the space
for i,layer_objects in enumerate(scene_objects):
  # Add the balls
  layer_objects['balls'] = []
  for circle in layer_objects['circles']:
    radius = circle[2]
    body = pymunk.Body()
    if i == ImgToObj.Layer.static_goal.value or i == ImgToObj.Layer.static_body.value:
      body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = circle[0],circle[1]
    shape = pymunk.Circle(body, radius, Vec2d(0,0))
    shape.elasticity = 0.3
    shape.friction = 0.6
    shape.density = 1
    if i == ImgToObj.Layer.object.value:
      shape.collision_type = COLLISION_OBJECT
    elif i == ImgToObj.Layer.dynamic_goal.value or i == ImgToObj.Layer.static_goal.value:
      shape.collision_type = COLLISION_GOAL
    space.add(body, shape)
    layer_objects['balls'].append((shape,[body.position]))

  # Add the polygons to the space
  # A list of lists of triangles per polygon
  layer_objects['pymunk_polys'] = []
  for polygon in layer_objects['polygons']:
    triangles = []
    center = np.mean(polygon[1]['vertices'].astype(float),axis=0)
    body = pymunk.Body()
    if i == ImgToObj.Layer.static_goal.value or i == ImgToObj.Layer.static_body.value:
      body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = center
    for triangle in polygon[1]['triangles'].tolist():
      shape = pymunk.Poly(body,polygon[1]['vertices'][triangle,:].astype(float)-center,radius=1.0)
      shape.elasticity = 0.3
      shape.friction = 0.6
      shape.density = 1
      if i == ImgToObj.Layer.object.value:
        shape.collision_type = COLLISION_OBJECT
      elif i == ImgToObj.Layer.dynamic_goal.value or i == ImgToObj.Layer.static_goal.value:
        shape.collision_type = COLLISION_GOAL
      space.add(shape)
      triangles.append(shape)
    space.add(body)
    layer_objects['pymunk_polys'].append(triangles)

dt = 1.0/60.0
time_sim = len(images)
num_steps = int(time_sim/dt)
steps_per_sec = int(num_steps/time_sim)

print("Seconds Simulated",time_sim)
print("Steps Simulated",num_steps)
print("Time Step",dt)

for i in range(num_steps):
  space.step(dt)
  # Delay fixed time between frames
  for layer_objects in scene_objects:
    for ballData in layer_objects['balls']:
      ballData[1].append(ballData[0].body.position)

  if i % steps_per_sec == 0:
    img_end = phyre.vis.observations_to_float_rgb(images[int(i/steps_per_sec)])
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img_end)
    axs[1].imshow(np.full_like(img_end,255))
    axs[0].title.set_text('{} sec out of {}'.format(int(i/steps_per_sec),time_sim))
    patches = []
    for layer_objects in scene_objects:
      for ballData in layer_objects['balls']:
        circle1=plt.Circle((ballData[0].body.position.x,256.0-ballData[0].body.position.y),radius=ballData[0].radius,color='b',fill=True)
        axs[1].add_artist(circle1)

      for pymunk_poly in layer_objects['pymunk_polys']:
        for triangle in pymunk_poly:
          verts = []
          for v in triangle.get_vertices():
            x,y = v.rotated(triangle.body.angle) + triangle.body.position
            verts.append([x,y])
          verts = np.array(verts)
          verts[:,1] = 256.0-verts[:,1]
          polygon = Polygon(verts,color='b')
          patches.append(polygon)

    p1 = PatchCollection(patches)
    p2 = PatchCollection(patches, alpha=0.4)
    axs[1].add_collection(p1)
    axs[0].add_collection(p2)
    plt.show()

t1 = time.time()
print(t1-t0,"Sim Time")
exit()


img_end = phyre.vis.observations_to_float_rgb(images[-1])
fig, axs = plt.subplots(1,2)
axs[0].imshow(img_end)
axs[1].imshow(np.full_like(img_end,255))


for i,ball in enumerate(balls):
  circle1=plt.Circle((ball.body.position.x,256.0-ball.body.position.y),radius=ball.radius,color='b',fill=True)
  axs[1].add_artist(circle1)
  ballTrail = np.array(ballPos[i])
  axs[1].plot(ballTrail[:,0],256.0-ballTrail[:,1])


patches = []

for triangle in triangles:
  verts = []
  for v in triangle.get_vertices():
    x,y = v.rotated(triangle.body.angle) + triangle.body.position
    verts.append([x,y])
  verts = np.array(verts)
  verts[:,1] = 256.0-verts[:,1]
  polygon = Polygon(verts,color='b')
  patches.append(polygon)
p = PatchCollection(patches, alpha=0.4)
axs[1].add_collection(p)

plt.show()

