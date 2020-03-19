import numpy as np
import phyre
import cv2
import polytri
from enum import Enum
import matplotlib.pyplot as plt

GRAV_PIX_PER_SEC = -30.0
FRAME_PER_SEC = 60.0

class Layer(Enum):
  background = 0
  action = 1
  object = 2
  dynamic_goal = 3
  static_goal = 4
  dynamic_body = 5
  static_body = 6


def phyreImgToBoolLayers(img):
  assert len(img.shape) == 2
  ncols = phyre.creator.constants.NUM_COLORS
  labels_one_hot = (img.ravel()[np.newaxis] ==
                    np.arange(ncols)[:, np.newaxis]).T
  labels_one_hot.shape = img.shape + (ncols,)
  return labels_one_hot

def phyreRadiusToPixelRadius(r):
  return r * 30.0 + 1.9

def pixelRadiusToPhyreRadius(r):
  return (r - 1.9)/30.0

def phyreActionToPixelAction(action):
  pixelAction = [0, 0, 0]
  pixelAction[0] = action[0] * 256.0
  pixelAction[1] = action[1] * 256.0
  pixelAction[2] = action[2] * 30.0 + 1.9
  return pixelAction

def pixelActionToPhyreAction(action):
  phyreAction = [0, 0, 0]
  phyreAction[0] = action[0]/256.0
  phyreAction[1] = action[1]/256.0
  phyreAction[2] = (action[2] - 1.9)/30.0
  return phyreAction


def isContourCircle(contour):
  M = cv2.moments(contour)
  circleMeasure = (M['m00'] * M['m00']) / (2*np.pi * (M['mu20']+M['mu02']))
  return circleMeasure > .95


def seperateCircleFromOtherContours(contours):
  circles = []
  nonCircles = []
  for contour in contours:
    contour = np.squeeze(contour)
    if cv2.contourArea(contour) > 0.0:
      if isContourCircle(contour):
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circles.append((x, y, radius))
      else:
        nonCircles.append(contour)
  return (nonCircles, circles)


def simplifyContours(contours):
  simpleContours = []
  for contour in contours:
    if contour.shape[0] > 4:
      simpleContours.append(np.squeeze(cv2.approxPolyDP(contour, 1.0, True)))
    else:
      simpleContours.append(contour)
  return simpleContours

# Turns the phyre scene image into geometric objects
# Returns a list of objects per layer as dict:
# 'raw_img': a grayscaled img of the layer
# 'circles': a list of all the circles in the layer as a tuple (x,y,radius)
# 'polygons': a list of all the polygons in the layer as a tuple of (n by 2 array of points,triangle array)


def phyreToObj(img, action=None):
  # Turn the labels in a one hot vector img
  layered_img = phyreImgToBoolLayers(img)

  # Extract the contours for each layer, not including the background layer
  scene_objects = []

  for i in range(layered_img.shape[2]):
    # Turn the boolean img into a greyscale img
    gray_layer = np.full_like(layered_img[:, :, i], 255, dtype=np.uint8)
    gray_layer = np.where(layered_img[:, :, i], gray_layer, 0)
    kernel = np.ones((3, 3), np.uint8)
    gray_layer = cv2.morphologyEx(gray_layer, cv2.MORPH_OPEN, kernel)
    circles = []
    polygons = []

    if i != Layer.background.value:
      # Extract the contours
      contours, hierarchy = cv2.findContours(
          gray_layer, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

      # Seperate circles from the contours
      contours, circles = seperateCircleFromOtherContours(contours)
      if i == Layer.action.value and action is not None:
        circles.append(phyreActionToPixelAction(action))

      # Simplify the non circle contours
      simple_contours = simplifyContours(contours)

      for contour in simple_contours:
        polytri_array = []
        try:
          polygon_tri = polytri.triangulate(contour)
          for triangle in polygon_tri:
            polytri_array.append(np.array(triangle))
        except ValueError:
          print("triangluation failed :(")
        polygons.append((contour, polytri_array))

    scene_objects.append(
        dict(raw_img=gray_layer, circles=circles, polygons=polygons))
  return scene_objects


def getObjectAndGoalSequence(img_seq):
  seq_info = {'object': [], 'goal': []}
  layer_names = ['object', 'goal']
  goal_type = Layer.dynamic_goal.value
  if goal_type not in img_seq[0]:
    goal_type = Layer.static_goal.value

  for img in img_seq:
    layered_img = phyreImgToBoolLayers(img)
    object_layer = layered_img[:, :, Layer.object.value]
    goal_layer = layered_img[:, :, goal_type]
    layers = [object_layer, goal_layer]

    for name, layer in zip(layer_names, layers):
      gray_layer = np.full_like(layer, 255, dtype=np.uint8)
      gray_layer = np.where(layer, gray_layer, 0)
      kernel = np.ones((3, 3), np.uint8)
      gray_layer = cv2.morphologyEx(gray_layer, cv2.MORPH_OPEN, kernel)
      contours, _ = cv2.findContours(
          gray_layer, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

      # Seperate circles from the contours
      contours, circles = seperateCircleFromOtherContours(contours)

      # Simplify the non circle contours
      contours = simplifyContours(contours)

      assert len(circles) <= 1, "More than one circle?"
      assert len(contours) <= 1, "More than one polygon?"

      if len(contours) > 0:
        seq_info[name].append({'type': "polygon", 'data': contours[0],
                               'centroid': np.mean(contours[0].astype(float), axis=0),
                               'bb': polygon_bounding_box(contours[0].astype(float))})
      elif len(circles) > 0:
        seq_info[name].append({'type': "circle", 'data': circles[0],
                               'centroid': circles[0][0:2],
                               'bb': circle_bounding_box(circles[0])})

  return seq_info


def objectTouchGoalSequence(img_seq):
  longest_sequence_len = 0
  current_sequence_len = 0
  goal_type = Layer.dynamic_goal.value
  if goal_type not in img_seq[0]:
    goal_type = Layer.static_goal.value
  for img in img_seq:
    if objectTouchGoal(img, goal_type):
      current_sequence_len += 1
      if longest_sequence_len < current_sequence_len:
        longest_sequence_len = current_sequence_len
    else:
      current_sequence_len = 0
    # print(current_sequence_len)
  return longest_sequence_len


def objectTouchGoal(img, goal_type):
  #fully_connected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
  cross = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
  layered_img = phyreImgToBoolLayers(img)
  object_layer = layered_img[:, :, Layer.object.value]
  gray_layer = np.full_like(object_layer, 255, dtype=np.uint8)
  gray_layer = np.where(object_layer, gray_layer, 0)
  gray_layer = cv2.dilate(gray_layer, cross, iterations=1)
  test_layer = np.where(
      layered_img[:, :, goal_type], gray_layer, np.zeros_like(gray_layer))
  '''
  if np.sum(test_layer > 0) > 0:
    fig, axs = plt.subplots(3,1)
    axs[0].imshow(object_layer)
    axs[1].imshow(gray_layer)
    axs[2].imshow(layered_img[:, :, goal_type])
    plt.show()
  '''
  #print(np.sum(test_layer > 0) > 0,np.sum(test_layer > 0))
  return np.sum(test_layer > 0) > 2


def circle_bounding_box(circle):
  x = circle[0]
  y = circle[1]
  r = circle[2]
  return np.array([(x-r, y+r), (x+r, y+r), (x+r, y-r), (x-r, y-r)])


def polygon_bounding_box(polygon):
  min_x = np.amin(polygon[:, 0])
  max_x = np.amax(polygon[:, 0])

  min_y = np.amin(polygon[:, 1])
  max_y = np.amax(polygon[:, 1])

  return np.array([(min_x, max_y), (max_x, max_y), (max_x, min_y), (min_x, min_y)])

def rect_intersect(rect1, rect2):
  l1 = rect1[0]
  r1 = rect1[2]
  l2 = rect2[0]
  r2 = rect2[2]
  # If one rectangle is on left side of other
  if(l1[0] > r2[0] or l2[0] > r1[0]):
    return False

  # If one rectangle is above other
  if(l1[1] < r2[1] or l2[1] < r1[1]):
    return False

  return True

def check_seq_action_intersect(seq_data,stride,goal_type,action):
  got_intersect = False
  x, y, r = phyreActionToPixelAction(action)
  found_intersect = False
  for frame_index, object_data, goal_data in zip(range(len(seq_data['object'])), seq_data['object'], seq_data['goal']):
    if found_intersect:
      break
    goal_bb = goal_data['bb']
    object_bb = object_data['bb']
    time = frame_index * stride / FRAME_PER_SEC
    y_time = max(r,y + 1.0/2.0 * GRAV_PIX_PER_SEC * time * time)
    test_action_bb = [(x-r, y+r), (x+r, y+r), (x+r, y-r), (x-r, y-r)]
    test_action_time_bb = [(x-r, y_time+r), (x+r, y_time+r), (x+r, y_time-r), (x-r, y_time-r)]
    if rect_intersect(object_bb, test_action_bb) or rect_intersect(object_bb, test_action_time_bb):
      got_intersect = True
    elif goal_type == Layer.dynamic_goal.value and (rect_intersect(goal_bb, test_action_bb) or rect_intersect(goal_bb, test_action_time_bb)):
      got_intersect = True
  return got_intersect
