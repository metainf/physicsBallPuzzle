import time

import numpy as np

import phyre

import ImgToObj

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def rect_intersect(rect1, rect2):
  l1 = rect1[0, :]
  r1 = rect1[2, :]
  l2 = rect2[0, :]
  r2 = rect2[2, :]
  # If one rectangle is on left side of other
  if(l1[0] > r2[0] or l2[0] > r1[0]):
    return False

  # If one rectangle is above other
  if(l1[1] < r2[1] or l2[1] < r1[1]):
    return False

  return True

def overlappingArea(rect1, rect2): 
  if rect_intersect(rect1,rect2):
    l1 = rect1[0]
    r1 = rect1[2]
    l2 = rect2[0]
    r2 = rect2[2]
    
    areaI = (min(r1[0], r2[0]) - max(l1[0], l2[0])) * (min(l1[1], l2[1]) - max(r1[1], r2[1])) 
    return areaI
  else:
    return 0

def rectArea(rect):
  l = rect[0]
  r = rect[2]
  area = abs(l[0] - r[0]) * abs(l[1] - r[1])
  return area

ball_sizes = np.linspace(0.01, 1, 5)
pos = np.linspace(0, 1, 50)

actions = np.array(np.meshgrid(pos, pos, ball_sizes)).T.reshape(-1, 3)

valid_actions = []

for action in actions:
  x, y, r = ImgToObj.phyreActionToPixelAction(action)
  action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, y-r), (x-r, y-r)])
  if np.min(action_bb[:,0]) > 0 and np.max(action_bb[:,0]) < 255 and np.min(action_bb[:,1]) > 0 and np.max(action_bb[:,1]) < 255:
    valid_actions.append(action)

actions = np.array(valid_actions)
print(actions.shape)

tier = 'ball'
task_str = '00000:000'

simulator = phyre.initialize_simulator([task_str], tier)

initial_scene = simulator.initial_scenes[0]
frame_data = ImgToObj.getObjectAndGoalSequence([initial_scene])
goal_type = ImgToObj.Layer.dynamic_goal.value
if goal_type not in initial_scene:
  goal_type = ImgToObj.Layer.static_goal.value

goal_data = frame_data['goal'][0]
object_data = frame_data['object'][0]

goal_bb = goal_data['bb']
goal_center = goal_data['centroid']
object_bb = object_data['bb']
object_center = object_data['centroid']

good_actions = []
t0 = time.time()

print(rectArea(object_bb))

for action in actions:
  x, y, r = ImgToObj.phyreActionToPixelAction(action)
  action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, 0), (x-r, 0)])
  if (goal_center[0] - object_center[0]) * (object_center[0] - x) > 0 and rect_intersect(object_bb, action_bb) and overlappingArea(object_bb, action_bb) >= rectArea(object_bb):
    tl = action_bb[0, :].astype(int)
    br = action_bb[2, :].astype(int)
    rect_img = initial_scene[br[1]:tl[1],tl[0]:br[0]]
    #if ImgToObj.Layer.object.value in rect_img:
    good_actions.append(action)
  elif goal_type == ImgToObj.Layer.dynamic_goal.value:
    if (object_center[0] - goal_center[0]) * (goal_center[0] - x) > 0 and ImgToObj.rect_intersect(goal_bb, action_bb) and overlappingArea(goal_bb, action_bb) >= rectArea(goal_bb):
      tl = action_bb[0, :].astype(int)
      br = action_bb[2, :].astype(int)
      rect_img = initial_scene[br[1]:tl[1],tl[0]:br[0]]
      #if goal_type in rect_img:
      good_actions.append(action)

t1 = time.time()
print(t1-t0,"Check Actions Time")

print(len(good_actions))

good_actions.sort(reverse=True, key=lambda x: x[2])

x, y, r = ImgToObj.phyreActionToPixelAction(good_actions[0])
print(x,y,r)
action_bb = np.array([(x-r, y+r), (x+r, y+r), (x+r, 0), (x-r, 0)]).astype(int)
print(overlappingArea(object_bb, action_bb))
tl = action_bb[0, :]
br = action_bb[2, :]

print(tl,br)

rect_img = initial_scene[br[1]:tl[1],tl[0]:br[0]]
plt.imshow(rect_img)
plt.show()