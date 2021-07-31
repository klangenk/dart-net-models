
from math import sqrt, atan2, pow, pi, sin, cos
import numpy as np

DOUBLE_BULLS_EYE = 50
BULLS_EYE = 25
TRIPLE = 3
DOUBLE = 2
NORMAL = 1
OUT = 0

RINGVALUES = [DOUBLE_BULLS_EYE, BULLS_EYE, NORMAL, TRIPLE, NORMAL, DOUBLE]
#RINGS = [0.04007514, 0.09392611, 0.5685567, 0.6280432, 0.9380087, 1]
#RINGS = [0.037352941176, 0.09352941176, 0.582352941, 0.62941176, 0.95294117, 1]
RINGS = [0.0411764705882, 0.0970588235, 0.57058823, 0.62941176, 0.94117647, 1]
RINGS_EXTENDED = [0] + RINGS + [1.25]
RING_BORDERS = [(RINGS_EXTENDED[i], RINGS_EXTENDED[i+1]) for i in range(0, len(RINGS_EXTENDED) - 1)]

ANGLEVALUES = [11, 14, 9, 12, 5, 20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11]
ANGLES = list(np.arange(-171., 200., 18.))
ANGLE_BORDERS = [(ANGLES[i], ANGLES[(i + 1) % len(ANGLES)]) for i in range(0, len(ANGLES) - 1)]

FIELD_BORDERS = [(y, x) for x in ANGLE_BORDERS for y in RING_BORDERS[2:-1]] + [(RING_BORDERS[0], None), (RING_BORDERS[1], None), (RING_BORDERS[-1], None)]

RING_OFFSET = 0.005
ANGLE_OFFSET = 0

def applyOffset(t, offset):
  return (t[0] + offset, t[1] - offset)

RING_BORDERS_V2 = [
  [(0, RINGS[0])],
  [(RINGS[0], RINGS[1])],
  [(RINGS[1], RINGS[2]), (RINGS[3], RINGS[4])],
  [(RINGS[2], RINGS[3])],
  [(RINGS[4], RINGS[5])],
  [(RINGS[5], 1.25)],
]

RING_BORDERS_V2 = [[applyOffset(t, RING_OFFSET) for t in x] for x in RING_BORDERS_V2]
ANGLE_BORDERS_V2 = [applyOffset((t[0] + 360 % 360, t[1] + 360 % 360), ANGLE_OFFSET) for t in ANGLE_BORDERS]

FIELD_BORDERS_V2 = [[(z, x) for z in y] for x in ANGLE_BORDERS_V2 for y in RING_BORDERS_V2[2:-1]] + [
  [(applyOffset(RING_BORDERS[0], RING_OFFSET), (0.0, 360.0))],
  [(applyOffset(RING_BORDERS[1], RING_OFFSET), (0.0, 360.0))],
  [(applyOffset(RING_BORDERS[-1], RING_OFFSET), (0.0, 360.0))]
]

FIELD_BORDERS_SLICES = [[((RINGS[1] + RING_OFFSET, 1.0 - RING_OFFSET), x)] for x in ANGLE_BORDERS_V2]

def getAngleValue (angle):
  global ANGLES, ANGLEVALUES
  i = 0
  while angle > ANGLES[i]:
    i += 1
  return ANGLEVALUES[i]


def getRingValue (radius):
  global RINGS, RINGVALUES, OUT
  i = 0
  while i < len(RINGS) and radius > RINGS[i]:
    i += 1
  return RINGVALUES[i] if i < len(RINGS) else OUT

def angleToPoint(angle, radius):
  rad = angle / 180.0 * pi
  return (sin(rad) * radius, cos(rad) * radius)

def getScore (x, y):
  global OUT, BULLS_EYE, DOUBLE_BULLS_EYE
  radius = sqrt(pow(x, 2) + pow(y, 2))
  ringValue = getRingValue(radius)
  if ringValue == OUT:
    return [0, 0]
  if ringValue == BULLS_EYE:
    return [25, 1]
  if ringValue == DOUBLE_BULLS_EYE:
    return [25, 2]
  angle = atan2(y, x) * 180 / pi
  return [getAngleValue(angle), ringValue]
