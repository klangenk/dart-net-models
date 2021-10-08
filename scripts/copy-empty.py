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
OFFSET = 37

path = "/home/kevin/Desktop"

files = [
  glob(f"{path}/0-*.png"),
  glob(f"{path}/1-*.png")
]
m0 = min(len(files[0]), len(files[1]))
m1 = max(len(files[0]), len(files[1]))
for i in range(0, m1):
  images = [
    Image.open(files[0][i % m0]),
    Image.open(files[1][i % m0])
  ]
  images = [image.resize((int(image.width * HEIGHT / image.height), HEIGHT)) for image in images]
  random.shuffle(images)
  result = Image.new('RGB', (2 * WIDTH, HEIGHT))
  result.paste(images[0], (int((WIDTH - images[0].width) / 2), 0))
  result.paste(images[1], (int(WIDTH + (WIDTH - images[1].width) / 2), 0))
  filename = f'/home/kevin/Projekte/DartNet/models/data/train/empty/empty#empty#empty_{i + OFFSET}.jpg'
  result.save(filename)
  print(filename)

