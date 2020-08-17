import numpy as np
import nodelib

DEBUG = False
# DEBUG = True


class region():
    def __init__(self, shape):
        self.space = np.zeros(shape, dtype=bool)
        self.area = 0
        self.node_list = []

    def add(self, p1,p2 = None):
        if p2 == None:
            x,y = p1
        else:
            x,y = p1,p2
        self.space[x,y] = True
        self.area += 1
        self.node_list.append((x,y))
        # print("({:2d},{:2d}),area = {:2d}".format(x,y,self.area))
    # def merge(self, region_new):
    #     self.space = np.logical_or(self.space, region_new.space)
    #     self.area = np.sum(self.space)
    def visited(self, pos):
        x,y = pos
        return self.space[x,y]

    
def getSegment(data, threshold = 25, delta = 4):

    if DEBUG:
        area = 0
    diff_down = np.abs(data[0:-1] - data[1:]) 
    diff_right = np.abs(data[:,0:-1] - data[:,1:])
    # print (diff_down)

    shape = (data.shape)
    size_x, size_y = shape

    seg_idx = 1
    seg_array = np.zeros(shape,dtype = "int64")

    # regions = []
    # regions.append(0)
    from queue import Queue


    firstMerge = False
    for x in range(size_x):
        for y in range(size_y):
            # if DEBUG:
            #     print("In ({:d},{:d})".format(x,y))

            # if y==size_y-1:
            #     print("{:d},{:d}".format(x,y))

            if seg_array[x,y] == 0:
                
                if DEBUG:
                    print("Seg region {:d}:".format(seg_idx), end = '\t')
                # q = queue.Queue(1000)
                # visited = np.zeros(shape, dtype=bool)
                
                reg = region(shape)
                # regions.append(reg)
                # front = Queue(maxsize = size_x*2)
                front = Queue()
                # area = 0
                reg.add(x,y)
                seg_array[x,y] = seg_idx
                # visited[x,y] = True
                front.put((x,y))
                while not front.empty():
                    nx,ny = front.get()
                    # print('(',end='')
                    # print(nx,end=',')
                    # print(ny,end=')\t')
                    near_nodes = nodelib.getNearNode(nx,ny,size_x, size_y)
                    for pos in near_nodes:
                        px,py = pos
                        # print('(',end='')
                        # print(px,end=',')
                        # print(py,end=')\t')

                        # a unvisited node
                        if seg_array[px,py] == 0 :

                            # it's connected
                            if nx==px:
                                if diff_right[px,min(ny,py)] < threshold:
                                    reg.add(px,py)
                                    front.put((px,py))
                                    seg_array[px,py] = seg_idx
                            elif ny == py:
                                if diff_down[min(nx,px), py] < threshold:
                                    reg.add(px,py)
                                    front.put((px,py))
                                    seg_array[px,py] = seg_idx
                            else :
                                sys.exit(333)

                    # print()


                if reg.area < delta :
                    if seg_idx == 1:
                        seg_idx += 1
                        firstMerge = True
                        # continue
                    # # merge to near region
                    # nearhood = np.logical_and(nodelib.getFrame(reg.space), seg_array!=0)
                    # # print(nearhood)
                    # nidx = np.bincount(seg_array[nearhood]).argmax()
                    # x, y = reg.node_list[0]
                    else:
                        if x==0:
                            nidx = seg_array[x,y-1]
                        else:
                            nidx = seg_array[x-1,y]
                        if nidx==0:
                            print("({:d},{:d})".format(x,y))
                            # print(seg_array[90,92])
                            print("[SEG] Wrong: region idx.")
                            ##############
                            return seg_array
                        for pos in reg.node_list:
                            nx,ny = pos
                            seg_array[nx,ny] = nidx
                    # seg_array[seg_array > ridx] -= 1
                    # seg_idx -= 1

                    if DEBUG:
                        print("merge to : {:d}".format(nidx), end = '\t')
                else:
                    # change color index
                    seg_idx += 1
                    if DEBUG:
                        print("\t", end = '\t')
                if DEBUG:
                    area += reg.area
                    print("area: " ,area)
    
    if firstMerge:
        pass
    # print (seg_array)

    # ridx = 1
    # while ridx-1 < len(regions):
    #     reg = regions[ridx-1]
    #     if reg.area < delta :
    #         nearhood = nodelib.getFrame(reg.space)
    #         nidx = np.bincount(seg_array[nearhood]).argmax()
    #         regions[nidx-1].merge(reg)
    #         regions.pop(ridx-1)
    #         seg_array[seg_array == ridx] = nidx
    #         seg_array[seg_array > ridx] -= 1
    #         ridx -= 1


    #     ridx += 1

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




