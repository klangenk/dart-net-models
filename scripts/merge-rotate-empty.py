import csv
import os
import cv2
from glob import glob
from PIL import Image
import random
import numpy as np

train = True

inPath = '/media/kevin/Daten/Pictures/Dart/*/*' 
outPath = f'/home/kevin/Projekte/DartNet/models/data_rotated/{"train" if train else "valid"}'

HEIGHT = 480


folders = [f for f in glob(inPath) if os.path.isdir(f) and os.path.exists(f"{f}/darts_check.csv")]

def square(image):
    width, height, channels = image.shape
    size = np.min((width, height))
    return image[(width - size) // 2:(width - size) // 2 + size, (height - size) // 2:(height - size) // 2 + size, :]

def transform(image, M):
    image = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_CUBIC)
    image = square(image)
    image = cv2.resize(image, (HEIGHT, HEIGHT), interpolation=cv2.INTER_CUBIC)
    return image



for folder in folders:
    date = os.path.basename(folder)
    category = os.path.basename(os.path.dirname(folder))
    out = f"{outPath}/{category}/{date}"
    try:
        os.makedirs(out)
    except: pass
    rotations = [np.loadtxt(f'{folder}/M-0.txt'), np.loadtxt(f'{folder}/M-1.txt')]
    
        [id, *darts] = row
        if id == 'id': continue
        files = [f'{folder}/{id}-0.jpg', f'{folder}/{id}-1.jpg'] #glob(f'{folder}/{id}-*')
        images = [cv2.imread(file) for file in files]
        images = [transform(image, rotations[i]) for i, image in enumerate(images)]
        random.shuffle(images)
        c = "#".join([d if d != '' else '0-1' for d in darts])
        filename = f"{out}/{c}_{id}.jpg"
        image = cv2.hconcat(images)
        cv2.imwrite(filename, image)
        print(filename)
