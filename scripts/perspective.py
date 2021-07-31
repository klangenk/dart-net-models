# import the necessary packages
import cv2
from PIL import Image, ImageOps
from lib.image.general import show
import numpy as np
from lib.score import getScore
from lib.transformation import transformPoints, transformImage
from collections import Counter
import os
from glob import glob
from lib.score import angleToPoint
import argparse
import math

parser = argparse.ArgumentParser(description='Find Darts')
parser.add_argument('imagePath', type=str,
                  help='path to image to calibrate')

args = parser.parse_args()
files = glob(args.imagePath)

ref = cv2.imread('../../data/ref3.jpg')

def dartCoords(x, y):
  #border = 157
  border = 0
  return ((x - ref.shape[0] / 2) / (ref.shape[0] / 2 - border), (y - ref.shape[1] / 2) /  (ref.shape[1] / 2 - border))

def imageCoords(x,y):
  global ref
  border = 0
  return (x * (ref.shape[0] / 2 - border) + (ref.shape[0] / 2), y * (ref.shape[1] / 2 - border) + (ref.shape[1] / 2))

step = 1

angles = [x for x in range(0, 360, step)]
points = [imageCoords(*angleToPoint(angle, 1)) for angle in angles]

def distance(x, y):
  return np.sum(np.power((np.array(x)-np.array(y)),2))

def calc_rotation():
  transformedPoints = transformPoints(points, M, inverse = True)
  dists = [distance(transformedPoints[i], transformedPoints[(i+1) % len(transformedPoints)]) for i in range(len(transformedPoints))]
  ys = [p[1] for p in transformedPoints]
  maxIndex = dists.index(max(dists))
  lowIndex = ys.index(max(ys))
  degrees = (lowIndex- maxIndex - 1) * step
  return degrees

for file in files:
  imageFullName = os.path.basename(file)
  dirname = os.path.dirname(file)
  filename, fileExtension = os.path.splitext(imageFullName)
  #img = cv2.imread(file)
  #M = np.loadtxt(f"{dirname}/{filename}.txt")
  #dst = transformImage(img,M, ref.shape)
  #transformedPoints = transformPoints(points, M, inverse = True)
  #dists = [distance(transformedPoints[i], transformedPoints[(i+1) % len(transformedPoints)]) for i in range(len(transformedPoints))]
  #ys = [p[1] for p in transformedPoints]
  #maxIndex = dists.index(max(dists))
  #lowIndex = ys.index(max(ys))
  #degrees = (lowIndex- maxIndex - 1) * step
  #print(maxIndex, lowIndex, degrees)
  outputFile = f"{dirname}/{filename}.jpg"
  ImageOps.exif_transpose(Image.open(file)).save(outputFile)
  #for i, point in enumerate(transformedPoints):
  #  (x, y) = point
  #  cv2.circle(img, (int(x), int(y)), 10, (255,0,0) if maxIndex == i else (0, 0, 255) if lowIndex == i else (0, 255, 0), -1)
  #show(img, name='dst', debug=False)
  #show(cv2.imread(outputFile), name='dst2', debug=False)
  #if cv2.waitKey(0) == 113: break
cv2.destroyAllWindows()
