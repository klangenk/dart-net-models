import cv2
import numpy as np
from ..transformation import calcTransformation

refImg = cv2.imread('./img/original.jpg')
refMask = np.zeros(refImg.shape, dtype = "uint8")
cv2.circle(refMask, (refImg.shape[0]//2, refImg.shape[1]//2), refImg.shape[0]//2, (255, 255, 255), -1)

def getMask(img):
  print (refImg.shape)
  print (img.shape)
  transformation = calcTransformation(refImg, img, refMask)
  return cv2.warpPerspective(refMask, transformation, (img.shape[1], img.shape[0]))