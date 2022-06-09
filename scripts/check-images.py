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

parser = argparse.ArgumentParser(description='Check Labels')
parser.add_argument('imagePath', type=str,
                  help='path to image to calibrate')

args = parser.parse_args()
files = glob(args.imagePath)

for i, file in enumerate(files):
  try:
    Image.open(file).load()
  except:
    print(file)
    os.unlink(file)
    pass
  if i % 1000 == 0: print(i)
