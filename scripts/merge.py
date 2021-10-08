import csv
import os
from glob import glob
from PIL import Image
import random

inPath = '/media/kevin/Daten/Pictures/Dart/*' 
#inPath = '/home/kevin/Projekte/DartNet/finder/img/*'
outPath = f'/home/kevin/deep-learning/data/train/esp32'

HEIGHT = 480


folders = [f for f in glob(inPath) if os.path.isdir(f) and os.path.exists(f"{f}/darts_check.csv")]

for folder in folders:
    out = f"{outPath}/{os.path.basename(folder)}"
    try:
        os.makedirs(out)
    except: pass
    with open(f'{folder}/darts_check.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            [id, *darts] = row
            if id == 'id': continue
            files = [f'{folder}/{id}-0.jpg', f'{folder}/{id}-1.jpg'] #glob(f'{folder}/{id}-*')
            images = [Image.open(file) for file in files]
            images = [image.rotate(90) if image.height > image.width else image for image in images]
            images = [image.resize((int(image.width * HEIGHT / image.height), HEIGHT)) for image in images]
            random.shuffle(images)

            width = max(*[image.width for image in images])
            
            result = Image.new('RGB', (len(images) * width, HEIGHT))
            w = 0
            for image in images:
                result.paste(image, (int(w + (width - image.width) / 2), 0))
                w += width
            c = "#".join([d if d != '' else 'empty' for d in darts])
            filename = f"{out}/{c}_{id}.jpg"
            result.save(filename)
            print(filename)
