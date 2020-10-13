import os, sys
from util import nodelib, imglib
from seg import seglib

import numpy as np
# import segment
# from util import imglib

# array0 = imglib.getImg("Pic/lena_gray_label.png", to_3d=True)
# print(type(array0))
# array0 = (np.random.rand(5,5)*20).astype(int)

# if i and j:
#     i+=1

# print(i)
# import random
# size = (15,20)
# posList = []
# num = 10
# for i in range(num):
#     ox,oy = random.randint(0,size[0]-1),random.randint(0,size[1]-1)
    

#     posList.append((ox,oy))

# print(sorted(posList))
# print(posList)
# import matplotlib.pyplot as plt
# i = 2
# dx = 0.5/i
# length = 6/dx
# sigma = float(sys.argv[1])
# x_axis = (np.arange((length*2+1))-length)*dx
# x_2 = np.square(x_axis)
# hx = (4*sigma**2 * x_2-2*sigma) * np.exp(-sigma*x_2)
# # print("W :" , np.mean(hx))
# hx -= np.mean(hx)
# # hx -= np.mean(hx)
# # if np.mean(hx)!=0:
# #     print("W :" , np.mean(hx))
# # x_axis = np.arange((length*2+1))-length
# # print(hx)
# plt.plot(x_axis,hx)
# plt.show()

color1 = np.array([[[115,82,68],[194,150,130],[98,122,157]]])
color2 = imglib.RGBtoLAB(color1)
print(color2)
for i in color1:
    for j in i:
        print(imglib.soleRGBtoLAB(j) )


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
