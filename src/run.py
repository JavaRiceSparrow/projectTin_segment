import numpy as np
import os, sys
# from array import array

import imglib
import nodelib
import color

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


threshold = 25
diff_down = np.abs(array1[0:-1] - array1[1:]) 
diff_right = np.abs(array1[:,0:-1] - array1[:,1:])
# print(np.mean(diff_down))
# print(np.mean(diff_right))


shape = (array1.shape)
# crs_arr = array('i')
# crs_arr.append(0)
# # crs_idx = 1

seg_idx = 1

size_x, size_y = shape

seg_array = np.zeros(shape)

# seg_array[0,0] = len(crs_arr)
# crs_arr.append(seg_idx)
# # crs_idx += 1
# seg_idx += 1
# import queue
for x in range(size_x):
    for y in range(size_y):
        if seg_array[x,y] == 0:
            # q = queue.Queue(1000)
            visited = np.zeros(shape, dtype=bool)
            path = []
            visited[x,y] = True
            path.append((x,y))
            while len(path) != 0:
                nx,ny = path.pop()
                path.append((nx,ny))
                isEndnode = True
                near_node = nodelib.getNearNode(nx,ny,size_x, size_y)
                for pos in near_node:
                    px,py = pos
                    # a unvisited node
                    if seg_array[px,py] == 0 and not visited[px,py]:

                        # it's connected
                        if nx==px:
                            if diff_right[px,min(ny,py)] < threshold:
                                visited[px,py] = True
                                isEndnode = False
                                path.append((px,py))
                        elif ny == py:
                            if diff_down[min(nx,px), py] < threshold:
                                visited[px,py] = True
                                isEndnode = False
                                path.append((px,py))
                        else :
                            sys.exit(333)

                # color and abandan it if it doesn't have branch
                if isEndnode:
                    seg_array[nx,ny] = seg_idx
                    path.pop()

            # change color index
            seg_idx += 1

# print(np.mean(seg_array))
# array1[edge] = [255,255,255]
segment = np.zeros(array0.shape)
for x in range(size_x):
    for y in range(size_y):
        segment[x,y] = np.array(color.getHue(seg_array[x,y]*10))

compare = imglib.mergeArray((array0, segment),axis=1, interval=20)
imglib.saveImg(compare, "output/1.gif")




edge = nodelib.getEdge(array1)               





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




