import numpy as np
import phyre
import cv2
import matplotlib.pyplot as plt
import triangle as tri
from enum import Enum

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
  labels_one_hot = (img.ravel()[np.newaxis] == np.arange(ncols)[:, np.newaxis]).T
  labels_one_hot.shape = img.shape + (ncols,)
  return labels_one_hot

def phyreActionToPixelAction(action):
  pixelAction = [0,0,0]
  pixelAction[0] = action[0] * 255.0
  pixelAction[1] = action[1] * 255.0
  pixelAction[2] = action[2] * 30.0 + 1.9
  return pixelAction

def isContourCircle(contour):
  M = cv2.moments(contour)
  circleMeasure = (M['m00'] * M['m00']) / (2*np.pi * (M['mu20']+M['mu02']))
  return circleMeasure > .95

def seperateCircleFromOtherContours(contours):
  circles = []
  nonCircles = []
  for contour in contours:
    contour = np.squeeze(contour)
    if isContourCircle(contour):
      (x,y),radius = cv2.minEnclosingCircle(contour)
      circles.append((x,y,radius))
    else:
      nonCircles.append(contour)
  return (nonCircles,circles)

def simplifyContours(contours):
  simpleContours = []
  for contour in contours:
    if contour.shape[0] > 4:
      simpleContours.append(np.squeeze(cv2.approxPolyDP(contour,0.5,True)))
    else:
      simpleContours.append(contour)
  return simpleContours

def triangulateContour(contour):
  verts = []
  segs = []
  for i in range(contour.shape[0]):
    verts.append((contour[i,0],contour[i,1]))
    segs.append((i,(i+1)%contour.shape[0]))
  A = dict(vertices=verts, segments=segs)
  B = tri.triangulate(A,'p')
  return B

# Turns the phyre scene image into geometric objects
# Returns a list of objects per layer as dict:
# 'raw_img': a grayscaled img of the layer
# 'circles': a list of all the circles in the layer as a tuple (x,y,radius)
# 'polygons': a list of all the polygons in the layer as a tuple of (n by 2 array of points,triangle dict)
def phyreToObj(img,action):
  # Turn the labels in a one hot vector img
  layered_img = phyreImgToBoolLayers(img)

  # Extract the contours for each layer, not including the background layer
  scene_objects = []

  for i in range(layered_img.shape[2]):
    # Turn the boolean img into a greyscale img
    gray_layer = np.full_like(layered_img[:,:,i],255,dtype=np.uint8)
    gray_layer = np.where(layered_img[:,:,i],gray_layer,0)
    circles = []
    polygons = []

    if i != Layer.background.value:
      # Extract the contours
      contours,hierarchy = cv2.findContours(gray_layer,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

      # Seperate circles from the contours
      contours,circles = seperateCircleFromOtherContours(contours)
      if i == Layer.action.value:
        circles.append(phyreActionToPixelAction(action))

      # Simplify the non circle contours
      simple_contours = contours #simplifyContours(contours)

      for contour in simple_contours:
        polygon_tri = triangulateContour(contour)
        polygons.append((contour,polygon_tri))

    scene_objects.append(dict(raw_img=gray_layer,circles=circles,polygons=polygons))
  return scene_objects






