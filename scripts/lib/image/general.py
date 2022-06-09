import cv2
import numpy as np
import math

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

def show (img, height = 1000, width = 1900, name='image', debug=True, clickHandler=None):
  w = min(width, img.shape[1] * height // img.shape[0])
  h = min(height, img.shape[0] * width // img.shape[1])
  #if (cv2.getWindowProperty(name, 0) <= 0):
  cv2.namedWindow(name,cv2.WINDOW_NORMAL)
  cv2.resizeWindow(name, w, h)
  if clickHandler is not None:
    def handler (event, x, y, flags, param):
      clickHandler(event, x , y,  flags, param)
    cv2.setMouseCallback(name, handler)
  
  cv2.imshow(name, cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC))
  if debug:
    cv2.imwrite('./debug/{0}.jpg'.format(name), img)
    cv2.waitKey(0)
  else:
    cv2.waitKey(1)

def triangleKernel(height, width):
  def row(width1, width2):
    return list(map(lambda x: 0 if abs(width1//2 - x) > (width2-1)//2 else 1 ,range(width1)))
  return np.array(list(map(lambda x: row(width, x), np.linspace(1, width, height))), dtype=np.uint8)

def erdil(img, kernelWidth=10, kernelHeight=None, kernelType=None):
  if kernelHeight is None:
    kernelHeight = kernelWidth
  kernel = None
  if (kernelType == 'triangle'):
    kernel = triangleKernel(kernelHeight, kernelWidth)
  else:
    kernel = np.ones((kernelHeight, kernelWidth), np.uint8)
  return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def diler(img, kernelSize=10, kernelType=None):
  kernel = None
  if (kernelType == 'triangle'):
    kernel = triangleKernel(kernelSize, kernelSize)
  else:
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
  
  return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def diff(emptyImg, actImg, mask=None, th=100, kernelSize=10):
  img = cv2.absdiff(emptyImg, actImg)
  if (mask is not None):
    img = cv2.bitwise_and(img,mask)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, img = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)
  img = erdil(img, kernelSize)


  kernel = np.ones((kernelSize,kernelSize),np.uint8)

  img = cv2.dilate(img, kernel)
  img = cv2.erode(img, kernel)

  return img

def diff3(img1, img2, mask=None):
  diff = cv2.absdiff(img1, img2)
  diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
  _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
  return cv2.bitwise_and(diff, diff, mask = mask)

def diff2(emptyImg, actImg):
  fgbg = cv2.createBackgroundSubtractorMOG2()
  fgbg.apply(emptyImg)
  return fgbg.apply(actImg)

def diffCount(emptyImg, actImg, mask=None, th=100, kernelSize=10):
  img = diff(emptyImg, actImg, mask)
  return cv2.countNonZero(img)



def findOuterPoints (diffImg, fromLeft):
  height, width = diffImg.shape
  outerPoints = [[0, height/2]]
  connected = False
  cols = range(0, width-1)
  offset = 1
  print (fromLeft)
  if not fromLeft:
    cols = reversed(cols)
    offset = -1

  
    
  for colIdx in cols:
    for rowIdx in range(0, height-1):
      if diffImg[rowIdx][colIdx] == 255:
        connected = True
        if diffImg[rowIdx][colIdx-offset] == 0 and  diffImg[rowIdx-1][colIdx-offset] == 0 and diffImg[rowIdx+1][colIdx-offset] == 0:
          if len([x for x in outerPoints if math.sqrt(pow(x[0]-colIdx, 2) + pow(x[1]-rowIdx, 2)) < 7]) == 0:
            outerPoints.append([colIdx, rowIdx])
              
      else:
        connected = False
  return outerPoints

def findOuterPoint (diffImg, direction):
  height, width = diffImg.shape
  cols = range(10, width-10)
  rows = range(10, height-10)
  nonzero = np.nonzero(diffImg)
  return [nonzero[1][0], nonzero[0][0]]

def drawPoints (img, points):
  result = None
  if (len(img.shape) < 3):
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  else:
    result = img.copy()
  
  imageHeight = result.shape[1]
  for [x,y] in points:
    cv2.circle(result, (int(x),int(y)), imageHeight//100, (0,0,255), imageHeight//500)
  return result

def concat(imgs):
  count = len(imgs)
  sqrt = math.sqrt(count)
  cols = math.ceil(sqrt)
  rows = math.ceil(count/cols)
  result = None

  blank = np.zeros(imgs[0].shape, np.uint8)
  
  for x in range(0, rows):
    temp = None
    for y in range(0, cols):
      index = x * cols + y
      img = imgs[index] if index < count else blank
      if len(img.shape) < 3 or img.shape[2] < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
      if temp is None:
        temp = img
      else:
        temp = np.hstack((temp, img))
    if result is None:
      result = temp
    else:
      result = np.vstack((result, temp))
  return result


def showMulti(imgs, height = 800, width = 1800 , name='image', debug=True):
  show(concat(imgs), height, width, name, debug)
