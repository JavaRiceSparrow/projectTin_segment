import numpy as np
from util import nodelib, imglib

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

    
def getSegment(data, threshold = 25, delta = 4, lamda = 0, edge=None):
    b_edge = False
    if lamda!=0:
        if type(edge) != np.ndarray:
            print("Edge is missed!")
            return None
        b_edge = True
        edge_array = edge*lamda
        

    array1 = imglib.img3dTo2d(data)

    if DEBUG:
        area = 0
    diff_down = np.abs(array1[0:-1] - array1[1:]) 
    diff_right = np.abs(array1[:,0:-1] - array1[:,1:])
    # if b_edge:
    #     diff_down = diff_down + edge[0:-1]
    #     diff_right = diff_right + edge[:,0:-1]

    # print (diff_down)

    shape = (array1.shape)
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
                           
                            if b_edge:
                                if nx==px:
                                    if diff_right[px,min(ny,py)] < threshold-edge_array[px,py]:
                                        reg.add(px,py)
                                        front.put((px,py))
                                        seg_array[px,py] = seg_idx
                                elif ny == py:
                                    if diff_down[min(nx,px), py] < threshold-edge_array[px,py]:
                                        reg.add(px,py)
                                        front.put((px,py))
                                        seg_array[px,py] = seg_idx
                                else :
                                    sys.exit(333)
                            else:
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


def getEdge(data):
    if len(data.shape) == 2:
        print("Wrong!")
        return None

    glist = []
    filter = [-1/6, -2/6, -3/6, 0, 3/6, 2/6, 1/6]
    
    for i in range(3):
        glist.append(matrixConvolution(data[:,:,i], filter, axis_along=0))
        glist.append(matrixConvolution(data[:,:,i], filter, axis_along=1))

    gsquare = np.square(glist.pop())
    while glist:
        gsquare = gsquare + np.square(glist.pop())

    return np.sqrt(gsquare)

def 

def getRidge(data)

def matrixConvolution(data, filter, axis_along = 0):
    if type(data) != np.ndarray:
        # print(type(data))
        print("Data type wrong!")
        return None
    if len(data.shape) != 2:
        print("Data size wrong!")
        # print(data.shape)
        return None
    filter = np.array(filter)
    # if type(filter) != np.ndarray:
    #     return None
    if len(filter.shape) != 1:
        # print("Data size wrong!")
        return None
    fsize = int(filter.shape[0])
    cdata = np.zeros((fsize,data.shape[0],data.shape[1]))
    mid = int((fsize-1)/2)
    if axis_along == 0:
        for i in range(mid):
            cdata[i,mid-i:] = data[:i-mid]

        cdata[mid] = data
        for i in range(mid):
            # i = i-mid
            cdata[-i-1][:i-mid] = data[mid-i:] 
    else :
        for i in range(mid):
            cdata[i,:,mid-i:] = data[:,:i-mid] 

        cdata[mid] = data
        for i in range(mid):
            # i = i-mid
            cdata[-i-1,:,:i-mid] = data[:,mid-i:] 
    
    for i in range(fsize):
        cdata[i] = cdata[i] * filter[i]
    
    return np.sum(cdata, axis = 0)


    
# '''
def processFile(data, b_seg_old=False):
    array0 = data
    
    # print("Shape: ", end="")
    # print(array0.shape)
    # array1 = imglib.img3dTo2d(array0)

    

    egde_array = getEdge(array0)
    
  
    seg_array_old = getSegment(array0, threshold=10, delta=8)
    # print("Seg time without edge:\t--- %8.4f seconds ---" % (time.time() - start_time))
    # start_time = time.time()
  
    seg_array = getSegment(array0, threshold=20, lamda = 0.1, edge=egde_array)


    # edata = imglib.arrToImg(255-egde_array/np.max(egde_array)*255)
    # segment_old = nodelib.toColor(seg_array_old)
    # segment = nodelib.toColor(seg_array)
    

    # try:
    #     DEF_SAVEIMG
    # except NameError:
    #     # not define
    #     pass
    # else:
    #     # define
    #     c1 = imglib.mergeArray((array0, segment_old),axis=1, interval=20)
    #     c2 = imglib.mergeArray((edata, segment),axis=1, interval=20)
    #     compare = imglib.mergeArray((c1,c2),axis=0, interval=20)
    #     imglib.saveImg(compare, out_path+path_name)

    if b_seg_old:
        return seg_array_old, egde_array, seg_array
    return egde_array, seg_array
# '''

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




