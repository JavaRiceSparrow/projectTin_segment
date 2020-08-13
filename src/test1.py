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

a = [3,4,5]
b = tuple(a)
print(a)
print(b)
print(a==b)