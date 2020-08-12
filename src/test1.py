# import os

# mypath = "Pic/"
# onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
# print(onlyfiles)


import numpy as np

a = np.array([[1,2],[3,4]])
b = np.around((a/3))
print(b)