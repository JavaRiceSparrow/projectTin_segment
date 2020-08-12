import numpy as np
import os, sys
from array import array

import imglib

path_name = ""


if len(sys.argv) != 2:
    print("No parameter!")
    os._exit(0)
    # if ()

path_name =  sys.argv[1] 

array1 = imglib.getImg(path_name, Black=False)
print("Shape: ", end="")
print(array1.shape)


diff_down = np.abs(array1[0:-1] - array1[1:])
diff_right = np.abs(array1[:,0:-1] - array1[:,1:])


shape = (array1.shape)
crs_arr = array('i')
crs_arr.append(0)
# crs_idx = 1
seg_idx = 1

size_x, size_y = shape
seg_array = np.zeros(shape)

seg_array[0,0] = len(crs_arr)
crs_arr.append(seg_idx)
# crs_idx += 1
seg_idx += 1

x = 1
for y in range(size_y-1):
    if diff_right[x,y] <= 25:
        seg_array[x,y+1] = seg_array[x,y]
    else:
        seg_array[x,y+1] = len(crs_arr)
        crs_arr.append(seg_idx)
        # crs_idx += 1
        seg_idx += 1
    
for x in range(1,size_x):
    for y in range(size_y):
        if diff_down[x-1,y] <= 25:
            seg_array[x,y] = seg_array[x-1,y]

    for y in range(size_y-1):
        if diff_right[x,y] <= 25:
            if seg_array[x,y+1] == 0:
                seg_array[x,y+1] = seg_array[x,y]
            else: 
                n1,n2 = seg_array[x,y+1], seg_array[x,y]
                if crs_arr[n1] != crs_arr[n2]:
                    crs_arr[n1] = crs_arr[n2]
        elif seg_array[x,y+1]:
            seg_array[x,y+1] = len(crs_arr)
            crs_arr.append(seg_idx)
            # crs_idx += 1
            seg_idx += 1




