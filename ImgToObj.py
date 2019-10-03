import numpy as np
import phyre
import cv2
import matplotlib.pyplot as plt

def phyreImgToBoolLayers(img):
  ncols = phyre.creator.constants.NUM_COLORS
  labels_one_hot = (img.ravel()[np.newaxis] == np.arange(ncols)[:, np.newaxis]).T
  labels_one_hot.shape = img.shape + (ncols,)
  return labels_one_hot

def findCirclesInPhyre(img):
  allCircles = []
  layeredImg = phyreImgToBoolLayers(img)
  for i in range(phyre.creator.constants.NUM_COLORS):
    circles = None
    if i > 0:
      grayImg = np.full_like(layeredImg[:,:,i],0,dtype=np.uint8)
      grayImg = np.where(layeredImg[:,:,i],grayImg,255)
      circles = cv2.HoughCircles(grayImg,cv2.HOUGH_GRADIENT,1,20,
              param1=15,
              param2=10)
      print(circles)
      #fig, ax = plt.subplots()
      #ax.imshow(grayImg)
      #ax.axis('off')
      #plt.show()
      #cv2.imshow('detected circles',grayImg)
      #cv2.waitKey(0)
    allCircles.append(circles)
  return allCircles
  
