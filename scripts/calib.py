# import the necessary packages
import cv2

from lib.image.general import show
import numpy as np
from glob import glob
from lib.score import getScore
from lib.transformation import transformPoints
from collections import Counter
import os
import argparse

dirname=""
filename=""

def drawPoint(img, x, y):
   cv2.circle(img, (x, y), 10, (0, 255, 0), 4)

def transformImage(img, M):
  return cv2.warpPerspective(img,M,(ref.shape[0],ref.shape[1]))

def generateHandler(pts, image, name):
  def click_and_crop(event, x, y, flags, param):
    global refPt, img, ref
    if event == cv2.EVENT_LBUTTONDOWN:
      pts.append((x, y))
      print(pts)
      cv2.circle(image, (x, y), 10, (0, 255, 0), 4)
      cv2.imshow(name, image)
      if len(pts) == 4:
        M = cv2.getPerspectiveTransform(np.float32(pts), np.float32(refPt))
        dst = transformImage(img,M)
        show(dst, name='dst', debug=False)
        np.savetxt(f"{dirname}/{filename}.txt", M)
        print(f"Saving {dirname}/{filename}.txt")
  return click_and_crop

def calib (img):
  imgPt = []
  show(img, height=1280, name='image', debug=False, clickHandler=generateHandler(imgPt, img, "image"))


parser = argparse.ArgumentParser(description='Calibrate')
parser.add_argument('imagePath', type=str,
                  help='path to image to calibrate')

args = parser.parse_args()

files = glob(args.imagePath)

refPt = [(1432, 19), (3377, 1435), (1964, 3379), (20, 1964)]

ref = cv2.imread('../../data_old/ref3.jpg')


for (x, y) in refPt:
  drawPoint(ref, x, y)



show(ref, height=720, name='ref', debug=False)
for file in files:
  imageFullName = os.path.basename(file)
  dirname = os.path.dirname(file)
  filename, fileExtension = os.path.splitext(imageFullName)
  img = cv2.imread(file)
  if img is None:
    print(f'Can not open {file}')
    continue
  calib(img)
  if cv2.waitKey(0) == 113: break
cv2.destroyAllWindows()
