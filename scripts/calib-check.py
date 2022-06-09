# import the necessary packages
import cv2

from lib.image.general import show
import numpy as np
from lib.score import getScore
from lib.transformation import transformPoints, transformImage
from collections import Counter
import os
from glob import glob
import argparse
parser = argparse.ArgumentParser(description='Find Darts')
parser.add_argument('imagePath', type=str,
                  help='path to image to calibrate')

args = parser.parse_args()
files = glob(args.imagePath)

ref = cv2.imread('/home/kevin/deep-learning/data_old/ref3.jpg')
#overlay = cv2.imread('../overlay3.png')


#print(ref.shape, overlay.shape)
for file in files:
  imageFullName = os.path.basename(file)
  dirname = os.path.dirname(file)
  filename, fileExtension = os.path.splitext(imageFullName)
  img = cv2.imread(file)
  M = np.loadtxt(f"{dirname}/{filename}.txt")
  dst = transformImage(img,M, ref.shape)
  #overlayed = cv2.addWeighted(dst,0.7,overlay,0.5,0)
  show(dst, name='dst', debug=False)
  if cv2.waitKey(0) == 113: break
cv2.destroyAllWindows()
