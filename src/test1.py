import os, sys
from util import nodelib, imglib
from seg import seglib

import numpy as np


color1 = np.array([[[115,82,68],[194,150,130],[98,122,157]]])

color2 = imglib.RGB2Lab(color1)
# cx = imglib.RGBtoXYZ(color1)
print(color2)
color2 = imglib.RGBtoLAB(color1)
# cx = imglib.RGBtoXYZ(color1)
print(color2)
# print(color2)
# for i in color1:
#     for j in i:
#         print(imglib.soleRGBtoLAB(j) )


# array1 = np.array([[1,1,1,2,3,3,3,3],[1,2,2,2,2,3,3,3],[2,2,2,2,2,4,4,3],\
#     [2,2,2,2,2,2,4,3],[2,2,5,5,5,5,5,5],[6,2,5,5,7,7,7,7]])


# array2[array2 == 1] = 250
# array2[array2 == 2] = 80
# array2[array2 == 3] = 150
# array2[array2 == 4] = 190
# array2[array2 == 5] = 120
# array2[array2 == 6] = 40
# array2[array2 == 7] = 250
# # print(array2)

# print(segment.getSegment(array2))
# f1 = np.logical_and(array1>3, array1<=5)

# print(array1[f1])

# a1 = np.array([3,3,3,4,4,4,5,5])
# print(np.bincount(a1).argmax())
