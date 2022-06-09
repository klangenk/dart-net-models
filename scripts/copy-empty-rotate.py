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
import shutil
import random

HEIGHT = 448
WIDTH = 800
OFFSET = 10

path = "/home/kevin/Projekte/DartNet/models/base_data_rotated/upper"
out = "/home/kevin/Projekte/DartNet/models/data_rotated/train/empty"


files = [
  glob(f"{path}/0/*/*.jpg"),
  glob(f"{path}/1/*/*.jpg")
]
m0 = len(files[0])
m1 = len(files[1])

print(m0, m1)


def square(image):
    width, height, channels = image.shape
    size = np.min((width, height))
    return image[(width - size) // 2:(width - size) // 2 + size, (height - size) // 2:(height - size) // 2 + size, :]

def transform(image, M):
    image = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_CUBIC)
    image = square(image)
    image = cv2.resize(image, (HEIGHT, HEIGHT), interpolation=cv2.INTER_CUBIC)
    return image

def read_rotation(file):
  folder = os.path.dirname(file)
  filename, extension = os.path.splitext(os.path.basename(file))
  try:
    return np.loadtxt(f'{folder}/M-{filename}.txt')
  except:
    suffix = filename.split('-')[-1]
    return np.loadtxt(f'{folder}/M-{suffix}.txt')

for i in range(0, m1):
  images = [
    transform(cv2.imread(files[0][i % m0]), read_rotation(files[0][i % m0])),
    transform(cv2.imread(files[1][i % m1]), read_rotation(files[1][i % m1]))
  ]
  
  random.shuffle(images)
  filename = f"empty#empty#empty_{i + OFFSET}.jpg"
  image = cv2.hconcat(images)
  cv2.imwrite(f"{out}/{filename}", image)
  print(filename)
