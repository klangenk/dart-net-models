# import the necessary packages
import cv2
from PIL import Image, ImageOps
from lib.image.general import RIGHT, show
import numpy as np
from lib.score import getScore
from lib.transformation import transformPoints, transformImage
from collections import Counter
import os
from glob import glob
import argparse
import math
import csv
from shutil import copyfile

parser = argparse.ArgumentParser(description='Find Darts')
parser.add_argument('imagePath', type=str,
                  help='path to images')

args = parser.parse_args()

SPACE = 32
D = 100
A = 97
ENTER = 13
RIGHT = 83
LEFT = 81
UP = 82
DOWN = 84
S = 115

def get_matrix(image, angle, tx = 0, ty = 0):
  translation_matrix = np.array([
      [0, 0, tx],
      [0, 0, ty]
  ], dtype=np.float32)
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  return rot_mat + translation_matrix

folders = sorted([f for f in glob(f'{args.imagePath}/*') if os.path.isdir(f)
  and os.path.exists(f"{f}/M-2.txt")
])

presets = [
  [0,0,0],
  [0,0,0],
  [0,0,0],
  [0,0,0]
]

for folder in folders:
  index = 3
  #files = sorted(glob(f'{folder}/*.jp*'), key=len)
  files = sorted(glob(f'{folder}/{index}-*'), key=len)
  preset_index = 0
  for file in files:
    suffix = os.path.basename(file).split('-')[1]
    original = cv2.imread(file)
    while True:
      preset = presets[preset_index]
      width, height, channels = original.shape
      size = np.min((width, height))
      M = get_matrix(original, preset[0], preset[1], preset[2])
      img = cv2.warpAffine(original, M, original.shape[1::-1], flags=cv2.INTER_LINEAR)
      img = img[(width - size) // 2:(width - size) // 2 + size, (height - size) // 2:(height - size) // 2 + size, :]
      show(img,debug=False)
      key = cv2.waitKey(0)
      print(key)
      if key == SPACE: break
      elif key == RIGHT: preset[1] += 1
      elif key == LEFT: preset[1] -= 1
      elif key == UP: preset[2] -= 1
      elif key == DOWN: preset[2] += 1
      elif key == D: preset[0] -= 5
      elif key == A: preset[0] += 5
      elif key == S: preset_index = (preset_index + 1) % len(presets)
      elif key == ENTER:
        np.savetxt(f"{folder}/M-{suffix}.txt", M)
        preset_index = (preset_index + 1) % len(presets)
        break
      else: exit()
