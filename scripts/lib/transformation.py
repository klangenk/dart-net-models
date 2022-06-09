import numpy as np
import cv2

def calcTransformation (fromImg, toImg, fromMask = None, toMask = None):
  #if (len(fromImg.shape) > 2):
  #  fromImg = cv2.cvtColor(fromImg, cv2.COLOR_BGR2GRAY)
  #if (len(toImg.shape) > 2):
  #  toImg = cv2.cvtColor(toImg, cv2.COLOR_BGR2GRAY)

  detector = cv2.BRISK_create()
  fromKeypoints, fromDescriptors = detector.detectAndCompute(fromImg,fromMask)
  toKeypoints, toDescriptors = detector.detectAndCompute(toImg,toMask)
  

  bf = cv2.BFMatcher()
  matches = bf.knnMatch(toDescriptors, fromDescriptors, k=2)

  points1 = []
  points2 = []
  for m,n in matches:
      if m.distance < 0.75*n.distance:
          points1.append(toKeypoints[m.queryIdx].pt)
          points2.append(fromKeypoints[m.trainIdx].pt)

  transformation, mask = cv2.findHomography(np.array(points2), np.array(points1), cv2.RANSAC, 5.0)
  return transformation

def transform (img, transformation, shape):
  cv2.warpPerspective(img, transformation, (shape[1], shape[0]))

def combine (*transformations):
  trans = list(transformations)
  
  trans.reverse()
  result = transformations[0]
  for t in transformations[1:]:
    result = np.dot(t, result)
  return result

def transformPoints(points, transformation, inverse = False):
  if inverse: transformation = np.linalg.pinv(transformation)
  [result] = cv2.perspectiveTransform(np.array([points], np.float32), transformation)
  return result

def transformImage(img, M, shape):
  return cv2.warpPerspective(img,M,(shape[0],shape[1]))

def unifyTransform (fromImg, toImg, fromMask = None, toMask = None):
  transformation = calcTransformation(fromImg, toImg, fromMask, toMask)
  return cv2.warpPerspective(fromImg, transformation, (toImg.shape[1], toImg.shape[0]))
