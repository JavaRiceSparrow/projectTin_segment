import os

# mypath = "Pic/"
# onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
# print(onlyfiles)


import numpy as np
# import segment
import imglib

array0 = imglib.getImg("Pic/lena_gray_label.png", to_3d=True)
print(array0.shape)


# a = np.array([[1,2],[3,4]])
# b = np.around((a/3))
# c = (a/3)

# print(a.dtype == 'int64')
# print(b.dtype == 'float64')
# print(c.dtype)

# array1 = np.array([[1,1,1,2,3,3,3,3],[1,2,2,2,2,3,3,3],[2,2,2,2,2,4,4,3],\
#     [2,2,2,2,2,2,4,3],[2,2,5,5,5,5,5,5],[6,2,5,5,7,7,7,7]])

# array2 = array1.copy()
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
