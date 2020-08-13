import numpy as np
import nodelib


def getSegment(data, threshold = 25):


    diff_down = np.abs(data[0:-1] - data[1:]) 
    diff_right = np.abs(data[:,0:-1] - data[:,1:])
    # print(np.mean(diff_down))
    # print(np.mean(diff_right))


    shape = (data.shape)
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

    return seg_array


# def deleteSmallSeg(data, delta = 4):
#     if len(data.shape) != 2:
#         return False
#     size_x, size_y = data.shape

#     maxIdx = np.max(data)
#     idx = 1

#     for x in range(size_x):
#         for y in range(size_y):
#             if data[x,y] == idx:

                


#                 idx += 1




