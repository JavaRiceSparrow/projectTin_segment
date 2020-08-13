# import os

# mypath = "Pic/"
# onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
# print(onlyfiles)


import numpy as np

# a = np.array([[1,2],[3,4]])
# b = np.around((a/3))
# c = (a/3)

# print(a.dtype == 'int64')
# print(b.dtype == 'float64')
# print(c.dtype)

array1 = np.array([[1,1,1,2,3,3,3,3],[1,2,2,2,2,3,3,3],[2,2,2,2,2,4,4,3],\
    [2,2,2,2,2,2,4,3],[2,2,5,5,5,5,5,5],[6,2,5,5,7,7,7,7]])

print(array1)
