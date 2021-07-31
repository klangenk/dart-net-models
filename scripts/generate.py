from PIL import Image, ImageDraw, ImageEnhance, ImageOps, ExifTags
import numpy as np
import cv2
import os
import math
from glob import glob
from lib.score import getScore, RING_BORDERS, ANGLE_BORDERS, angleToPoint, FIELD_BORDERS, FIELD_BORDERS, FIELD_BORDERS_V2, FIELD_BORDERS_SLICES
from lib.transformation import transformPoints
from collections import Counter
import argparse
import random
from multiprocessing import Pool
import time
import tqdm

IMAGE_HEIGHT = 448
IMAGE_WIDTH = 798
BOARDS_DIR = '../../data_old/boards/**/*.jpg'
DARTS_DIR = "../../data_old/darts/foo/2/*.png"
# '../data/sequential-4/train'
# '/mnt/c/Users/kevin/deep-learning/train'
OUTPUT_DIR = '/home/kevin/deep-learning/data/train/generated7'
SAMPLE_COUNT = 1

# load the image, clone it, and setup the mouse callback function
ref = cv2.imread('../../data_old/ref3.jpg')

try:
    os.makedirs(OUTPUT_DIR)
except: pass

def get_exif_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = image._getexif()

        if exif[orientation] == 3:
            return 180
        elif exif[orientation] == 6:
            return 270
        elif exif[orientation] == 8:
            return 90
        return 0
    except:
        # cases: image don't have getexif
        return 0


def dartCoords(x, y):
    border = 0
    return ((x - ref.shape[0] / 2) / (ref.shape[0] / 2 - border), (y - ref.shape[1] / 2) / (ref.shape[1] / 2 - border))


def imageCoords(x, y):
    border = 0
    return (x * (ref.shape[0] / 2 - border) + (ref.shape[0] / 2), y * (ref.shape[1] / 2 - border) + (ref.shape[1] / 2))


def rotate_around(point, degrees, origin=(0, 0)):
    radians = math.radians(degrees)
    (x, y) = point
    (ox, oy) = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return (int(qx), int(qy))


dartImages = [Image.open(file).rotate(-90) for file in glob(DARTS_DIR)]
dartCount = len(dartImages)
print(dartCount)

def my_random(a, b):
  low = min(a,b)
  high = max(a, b)
  if low == 0 and high == 360: return random.uniform(low,high)
  mid = (a + b) / 2
  r = random.gauss(mid, (high - low) / 5)
  x = mid - r
  if (x < 0): x = high + x
  else: x = low + x
  return min(high, max(low, x))


class RandomDart:
    def __init__(self, boardSize, rotation, index=None):
        self.index = random.randint(
            0, dartCount - 1) if index is None else index
        self.size = random.randint(
            280 * boardSize[1] // 1080, 320 * boardSize[1] // 1080)
        self.rotation = random.randint(-20, 20) - rotation
        self.offset = rotate_around(
            (self.size // 2, self.size), self.rotation, (self.size // 2, self.size // 2))
        self.contrast = random.uniform(0.3, 1.7)

    @property
    def v(self):
        return [self.index, self.size, self.rotation, self.contrast, *self.offset]


def getDartPoint(radiusBorders=(0, 1.25), angleBorders=(0.0, 360.0)):
    if radiusBorders is None:
        radiusBorders = (0, 1.25)
    if angleBorders is None:
        angleBorders = (0.0, 360.0)
    radius = my_random(*radiusBorders)
    angle = my_random(angleBorders[0] + 2 * (1 - radius / 1.2), angleBorders[1] - 2 * (1 - radius / 1.2))
    return angleToPoint(angle, radius)

def rasterize(count):
    result = []
    for x in FIELD_BORDERS_V2 + [None]:
        for y in FIELD_BORDERS_V2 + [None]:
            for z in FIELD_BORDERS_V2 + [None]:
                for _ in range(0, max(count, 1)):
                    if count < 1 and random.uniform(0, 1) > count:
                        continue
                    result.append([x,y,z])
    return result


def assignDartsAndBoards(files, example):
    boardIndex0 = random.randint(0, len(files) - 1)
    boardIndex1 = random.randint(0, len(files) - 1)
    while abs(files[boardIndex0][2] - files[boardIndex1][2]) < 15:
        boardIndex1 = random.randint(0, len(files) - 1)
    result0 = []
    result1 = []
    for index, x in enumerate(example):
        if x is None:
            result0.append(None)
            result1.append(None)
            continue
        coords = imageCoords(*x)
        transformedCoords0 = transformPoints(
            [coords], files[boardIndex0][1], inverse=True)
        dart0 = RandomDart(files[boardIndex0][0].size, files[boardIndex0][2])
        dart1 = RandomDart(files[boardIndex0][0].size, files[boardIndex1][2], dart0.index)
        result0.append([*dart0.v,
                        *transformedCoords0[0],  *getScore(*x), *x])
        transformedCoords1 = transformPoints(
            [coords], files[boardIndex1][1], inverse=True)
        result1.append([*dart1.v,
                        *transformedCoords1[0],  *getScore(*x), *x])

    return (boardIndex0, boardIndex1, result0, result1)


def loadTransform(file):
    imageFullName = os.path.basename(file)
    dirname = os.path.dirname(file)
    imageFilename, fileExtension = os.path.splitext(imageFullName)
    return np.loadtxt(f"{dirname}/{imageFilename}.txt")


def loadFile(file):
    img = Image.open(file)
    trans = loadTransform(file)
    rot = calc_rotation(trans)
    return (img, trans, rot)

def loadFiles(imagePath):
    return [loadFile(file) for file in glob(imagePath, recursive=True)]


cnt = Counter()


def prepare(sample):
    global cnt
    [boardIndex0, boardIndex1, darts0, darts1] = sample
    scores = [s and (s[8], s[9]) for s in darts0]
    positions = [s and (s[10], s[11]) for s in darts0]
    #scores.sort(key=lambda x: f"{x[0]}-{x[1]}" if x is not None else "empty")
    scoreLabel = '#'.join(
        [f"{score[0]}-{score[1]}" if score is not None else "empty" for score in scores]
    )
    positionLabel = '#'.join(
        [f"{pos[0]}#{pos[1]}" if pos is not None else "empty" for pos in positions]
    )
    label = f"{scoreLabel}${positionLabel}"
    cnt[label] += 1
    return (boardIndex0, boardIndex1, darts0, darts1, label, cnt[label])


def drawDarts(file, rotation, darts):
    indices = sorted(filter(lambda x: darts[x] is not None, range(
        0, len(darts))), key=lambda x: darts[x][7])
    current = file.copy()
    for index in indices:
        [dartIndex, dartSize, dartRotation, dartContrast,
            dartOffsetX, dartOffsetY, x, y, *rest] = darts[index]
        dartImage = ImageEnhance.Contrast(dartImages[dartIndex].copy().resize(
            (dartSize, dartSize), Image.ANTIALIAS).rotate(dartRotation)).enhance(dartContrast)
        current.paste(dartImage, (int(x) - dartOffsetX,
                                  int(y) - dartOffsetY), dartImage)
    return current

def saveImage(c, counter, darts0, darts1, outputFolder):
    filename = f"{outputFolder}/{c}_{counter}-2.jpg"
    d0 = darts0.resize((darts0.width * IMAGE_HEIGHT // darts0.height,
                        IMAGE_HEIGHT), Image.ANTIALIAS)
    d1 = darts1.resize((darts1.width * IMAGE_HEIGHT // darts1.height,
                        IMAGE_HEIGHT), Image.ANTIALIAS)
    result = Image.new('RGB', (IMAGE_WIDTH * 2, IMAGE_HEIGHT), (0, 0, 0, 0))
    if random.choice([True, False]):
        result.paste(d0, ((IMAGE_WIDTH - d0.width) // 2, 0))
        result.paste(d1, (IMAGE_WIDTH + (IMAGE_WIDTH - d1.width) // 2, 0))
    else:
        result.paste(d0, (IMAGE_WIDTH + (IMAGE_WIDTH - d0.width) // 2, 0))
        result.paste(d1, ((IMAGE_WIDTH - d1.width) // 2, 0))
    result.save(filename)


def init():
    global files, dartImages
    dartImages = [Image.open(file).rotate(-90) for file in glob(DARTS_DIR, recursive=True)]
    files = loadFiles(BOARDS_DIR)

bar = None

step = 1

angles = [x for x in range(0, 360, step)]
points = [imageCoords(*angleToPoint(angle, 1)) for angle in angles]

def distance(x, y):
  return np.sum(np.power((np.array(x)-np.array(y)),2))

def calc_rotation(M):
  transformedPoints = transformPoints(points, M, inverse = True)
  dists = [distance(transformedPoints[i], transformedPoints[(i+1) % len(transformedPoints)]) for i in range(len(transformedPoints))]
  ys = [p[1] for p in transformedPoints]
  maxIndex = dists.index(max(dists))
  lowIndex = ys.index(max(ys))
  degrees = (lowIndex- maxIndex - 1) * step
  return degrees

def handleSample(sample):
    global files, sampleSize, bar
    [x,y,z] = sample
    points = [
        x and getDartPoint(*random.choice(x)),
        y and getDartPoint(*random.choice(y)),
        z and getDartPoint(*random.choice(z))
    ]
    item = prepare(assignDartsAndBoards(files, points))
    board0 = item[0]
    board1 = item[1]
    d0 = drawDarts(files[board0][0], files[board0][2], item[2])
    d1 = drawDarts(files[board1][0], files[board1][2], item[3])
    saveImage(item[4], item[5], d0, d1, OUTPUT_DIR)
    #print(f"saving {i}/{sampleSize}")

if __name__ == "__main__":
    t = time.process_time()
    files = loadFiles(BOARDS_DIR)
    print(len(files))
    samples = rasterize(SAMPLE_COUNT)
    sampleSize = len(samples)
    pool = Pool(16, initializer=init)
    for x in tqdm.tqdm(pool.imap_unordered(handleSample, samples, 100), total=sampleSize):
        pass
    elapsed_time = time.process_time() - t
    print(f"created {sampleSize} files in {elapsed_time} minutes")
