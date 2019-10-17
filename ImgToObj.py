import numpy as np
import phyre
import cv2
import matplotlib.pyplot as plt

def phyreImgToBoolLayers(img):
  ncols = phyre.creator.constants.NUM_COLORS
  labels_one_hot = (img.ravel()[np.newaxis] == np.arange(ncols)[:, np.newaxis]).T
  labels_one_hot.shape = img.shape + (ncols,)
  return labels_one_hot

def findContoursInPhyre(img):
  allContours = []
  layeredImg = phyreImgToBoolLayers(img)
  for i in range(phyre.creator.constants.NUM_COLORS):
    contours = []
    if i > 0:
      grayImg = np.full_like(layeredImg[:,:,i],255,dtype=np.uint8)
      grayImg = np.where(layeredImg[:,:,i],grayImg,0)
      contours,hierarchy = cv2.findContours(grayImg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    allContours.append(contours)
  return allContours

def isContourCircle(contour):
  M = cv2.moments(contour)
  circleMeasure = (M['m00'] * M['m00']) / (2*np.pi * (M['mu20']+M['mu02']))
  return circleMeasure > .95

def seperateCircleFromOtherContours(contours):
  circles = []
  otherContours = []
  for contour in contours:
    if isContourCircle(contour):
      (x,y),radius = cv2.minEnclosingCircle(contour)
      circles.append((x,y,radius))
    else:
      otherContours.append(contour)
  return (otherContours,circles)

def simplifyContours(contours):
  simpleContours = []
  for contour in contours:
    minLength = np.linalg.norm(contour[0,:,:] - contour[-1,:,:])
    for i in range(contour.shape[0]-1):
      length = np.linalg.norm(contour[i,:,:] - contour[i+1,:,:])

    epsilon = 0.025*cv2.arcLength(contour,True)
    simpleContours.append(cv2.approxPolyDP(contour,epsilon,True))
    print(contour.shape,cv2.approxPolyDP(contour,epsilon,True).shape)
  return simpleContours

  
