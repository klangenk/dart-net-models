import csv
import os
from glob import glob
from PIL import Image
import random

folder = '/home/kevin/Pictures/2021-02-22-13-06'
out = f'{folder}/merged'

WIDTH = 800
HEIGHT = 448

try:
    os.makedirs(out)
except: pass

with open(f'{folder}/darts.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        [id, *darts] = row
        if id == 'id': continue
        files = glob(f'{folder}/{id}-*')
        images = [Image.open(file) for file in files]
        images = [image.rotate(90) if image.height > image.width else image for image in images]
        images = [image.resize((int(image.width * HEIGHT / image.height), HEIGHT)) for image in images]
        random.shuffle(images)
        i = 1
        for i1 in range(0, len(images)):
            for i2 in range(i1 + 1, len(images)):
                result = Image.new('RGB', (2 * WIDTH, HEIGHT))
                w = 0
                result.paste(images[i1], (int((WIDTH - images[i1].width) / 2), 0))
                result.paste(images[i2], (int(WIDTH + (WIDTH - images[i2].width) / 2), 0))
                c = "#".join([d if d != '' else 'empty' for d in darts])
                filename = f"{out}/{c}_{id}-{i}.jpg"
                result.save(filename)
                print(filename)
                i += 1
