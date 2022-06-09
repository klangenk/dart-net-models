from glob import glob
import os
import argparse
from shutil import copyfile

dirname=""
filename=""

parser = argparse.ArgumentParser(description='Calibrate')
parser.add_argument('imagePath', type=str,
                  help='path to images to calibrate')
parser.add_argument('calibration', type=str,
                  help='path to calibration')

args = parser.parse_args()

files = glob(args.imagePath)

for file in files:
  imageFullName = os.path.basename(file)
  dirname = os.path.dirname(file)
  filename, fileExtension = os.path.splitext(imageFullName)
  if args.calibration == f'{dirname}/{filename}.txt': continue
  copyfile(args.calibration, f'{dirname}/{filename}.txt')
