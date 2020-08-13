import numpy as np
import os, sys
# from array import array

import imglib
import color
from segment import *
import nodelib

path_name = ""


if len(sys.argv) != 2:
    print("No parameter!")
    os._exit(0)
    # if ()

path_name =  sys.argv[1] 

array0 = imglib.getImg(path_name, to_3d=True)
# print("Shape: ", end="")
# print(array1.shape)
array1 = imglib.img3dTo2d(array0)

seg_array = getSegment(array1, threshold=10)


# print(np.mean(seg_array))
# array1[edge] = [255,255,255]
size_x, size_y = array1.shape
segment = np.zeros(array0.shape)
for x in range(size_x):
    for y in range(size_y):
        segment[x,y] = np.array(color.getHue(seg_array[x,y]*10))

compare = imglib.mergeArray((array0, segment),axis=1, interval=20)
imglib.saveImg(compare, "output/3.gif")


# edge = nodelib.getEdge(array1)               

# x = 1
# for y in range(size_y-1):
#     if diff_right[x,y] <= 25:
#         seg_array[x,y+1] = seg_array[x,y]
#     else:
#         seg_array[x,y+1] = len(crs_arr)
#         crs_arr.append(seg_idx)
#         # crs_idx += 1
#         seg_idx += 1
    
# for x in range(1,size_x):
#     for y in range(size_y):
#         if diff_down[x-1,y] <= 25:
#             seg_array[x,y] = seg_array[x-1,y]

#     for y in range(size_y-1):
#         if diff_right[x,y] <= 25:
#             if seg_array[x,y+1] == 0:
#                 seg_array[x,y+1] = seg_array[x,y]
#             else: 
#                 n1,n2 = seg_array[x,y+1], seg_array[x,y]
#                 if crs_arr[n1] != crs_arr[n2]:
#                     crs_arr[n1] = crs_arr[n2]
#         elif seg_array[x,y+1]:
#             seg_array[x,y+1] = len(crs_arr)
#             crs_arr.append(seg_idx)
#             # crs_idx += 1
#             seg_idx += 1




