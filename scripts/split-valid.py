import csv
import os
from glob import glob
from PIL import Image
import random

inPath = '/media/kevin/Daten/Pictures/Dart/*/*' 

FACTOR = 0.1


folders = [f for f in glob(inPath) if os.path.isdir(f) and os.path.exists(f"{f}/darts_check.csv")]
print(f"date,id")
for folder in folders:
    date = os.path.basename(folder)
    with open(f'{folder}/darts_check.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            [id, *darts] = row
            if id == 'id': continue
            if random.random() < FACTOR:
                print(f"{date},{id}")
