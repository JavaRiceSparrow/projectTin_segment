import os, sys
from util import nodelib, imglib
from seg import seglib

import numpy as np



color1 = np.array([[115,82,68],[194,150,130],[98,122,157]])


def getRegDFS(region,node,b_list=False, b_area=False):
    x,y = node
    value = region[x,y]
    size = region.shape

    reg2 = np.zeros(region.shape,dtype=bool)
    reg2[x,y] = True
    if b_list:
        regList = [node]
    area = 1

    pos_stack=[node]
    # del node
    while(len(pos_stack) !=0):
        node1 = pos_stack.pop()
        nx,ny = node1
        nList = nodelib.getNearNode(nx,ny,size[0],size[1])
        for node2 in nList:
            if not reg2[node2] and region[node2]==value:
                pos_stack.append(node2)
                reg2[node2] = True
                area+=1
                if b_list:
                    regList.append(node2)

        del nList, node1, nx,ny
    if not b_list:
        if not b_area:
            return reg2
        else:
            return reg2, area
    else:
        # print(regList)
        if not b_area:
            return regList
        else:
            return regList, area

array1 = np.array([[1,1,1,2,3,3,3,3],[1,2,2,2,2,3,3,3],[2,2,2,2,2,4,4,3],\
    [2,2,2,2,2,2,4,3],[2,2,5,5,5,5,5,5],[6,2,5,5,7,7,7,7]])

plist = getRegDFS(array1,(2,2), True)

print(array1)
print(plist)
a2 = np.zeros(array1.shape)
for p in plist:
    a2[p] = 1
print(a2)
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
