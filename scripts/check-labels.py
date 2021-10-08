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
import argparse
import math
import csv
from shutil import copyfile

HEIGHT = 480

parser = argparse.ArgumentParser(description='Find Darts')
parser.add_argument('imagePath', type=str,
                  help='path to images')

args = parser.parse_args()

def resize(img, size):
  width, height, channels = img.shape
  if width == size[0] and height == size[1]:
    return img
  return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def vconcat_resize_min(im_list):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])))
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min))
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_tile_resize(im_list_2d):
    im_list_v = [hconcat_resize_min(im_list_h) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v)

def drawText(img, text, index):
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontScale = 1
  color = (255, 0, 0)
  thickness = 2
  
  textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
  pos = ((img.shape[1] - textsize[0]) // 2, (img.shape[0] + textsize[1]) // 2 + (index - 1) * 50)
  return cv2.putText(img, text, pos, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

def square(image):
  width, height, channels = image.shape
  size = np.min((width, height))
  return image[(width - size) // 2:(width - size) // 2 + size, (height - size) // 2:(height - size) // 2 + size, :]

def getSuffix(file):
  suffix = os.path.splitext(os.path.basename(file))[0].split('-')[1]
  return suffix

def readImage(file, transforms):
  suffix = getSuffix(file)
  img = cv2.imread(file)
  img = cv2.warpAffine(img, transforms[suffix], img.shape[1::-1], flags=cv2.INTER_CUBIC)
  img = square(img)
  return resize(img, (HEIGHT, HEIGHT))

SPACE = 32
D = 100
ENTER = 13

folders = sorted([f for f in glob(f'{args.imagePath}/*') if os.path.isdir(f)
  and os.path.exists(f"{f}/darts.csv")
  and not os.path.exists(f"{f}/darts_check.csv")
  and len(glob(f'{f}/*')) > 5
])



for folder in folders:
  with open(f'{folder}/darts.csv', newline='') as inFile:
    spamreader = csv.reader(inFile, delimiter=',', quotechar='"')
    with open(f'{folder}/darts_check.csv', 'w', newline='') as outFile:
      spamwriter = csv.writer(outFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      error = False
      partOf = False
      last = []
      lastCorrect = []
      transforms = {}
      for file in glob(f'{folder}/M-*.txt'):
        suffix = getSuffix(file)
        transforms[suffix] = np.loadtxt(file)
      for row in spamreader:
          [id, *darts] = row
          if id == 'id': 
            spamwriter.writerow(row)
            continue
          print(folder, id)
          partOf = darts[:len(last)] == last
          if error and partOf:
            for i, d in enumerate(lastCorrect):
              darts[i] = d
          files = sorted(glob(f'{folder}/{id}-*'))
          imgs = [readImage(file, transforms) for i, file in enumerate(files)]
          img = None
          if len(files) > 0 and all([i is not None for i in imgs]):
            h_min = min(im.shape[0] for im in imgs)
            if len(imgs) < 3:
              img = concat_tile_resize([imgs[:2]])
            else:
              img = concat_tile_resize([imgs[:2], imgs[2:]])
          else:
            print('Skipping', id)
            continue
          for i, dart in enumerate(darts):
            img = drawText(img, dart, i)
          show(img,debug=False)
          key = cv2.waitKey(0)
          error = key == SPACE or (error and partOf)
          if key == SPACE:
            last = list(filter(lambda x: x != '', darts))
            print(f'{",".join(last)}:')
            lastCorrect = input().split(",")
            line = lastCorrect + [""]*(3 - len(lastCorrect))
            spamwriter.writerow([id, *line])
          elif key == D: continue
          elif key == ENTER: spamwriter.writerow([id, *darts])
          else: exit()
