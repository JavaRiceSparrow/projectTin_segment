import numpy as np
import random 
import sys
# sys.path.append(".")
from util import nodelib, imglib
from seg import seglib
from seg.region import *
import time

in_path_str = "Pic/"

DEBUG = False
# DEF_LDATA = False #True

class Direction(object):
    #      -X
    #       ^
    #       |
    # -Y<---O---> Y
    #       |
    #       V
    #       X
    # 1 2 3
    # 0 X 4
    # 7 6 5
    
    dirList = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]

    def __init__(self, direIdx = 0):
        self.dir = direIdx
        
        
    def pos(self):
        return self.dirList[self.dir]
    def getEndPos(self, pos):
        x,y = pos
        o_pos = self.pos()
        return (x+o_pos[0],y+o_pos[1])
    def getDire(self):
        return self.dir
    def rotate(self, direc = True, time = 1):
        # print("[time] {:d}".format(time))
        if time != 1:
            for _ in range(time%8):
                self.rotate(direc = direc)
            return
        if direc:
            self.dir += 1
            if self.dir >= 8:
                self.dir -= 8
        else:
            self.dir -= 1
            if self.dir < 0:
                self.dir += 8
        # print("[time] end.")
    def rotateRev(self):
        self.dir += 4
        if self.dir >= 8:
            self.dir -= 8
        



def toMean(dataMgr, drawEdge = False):
    # print(dataMgr.shape)
    size_x, size_y = dataMgr.shape
    reg = dataMgr.region

    data = dataMgr.data_copy()
    out = np.zeros((size_x, size_y, 3))
    count = 0
    for i in range(reg.regSize):
        idx = i+1
        if reg.regSumList[idx] == 0:
            continue
        count += 1
        if count > reg.regNum:
            print("?")
        # if np.sum(dataMgr.regionBList[idx]) == 0:
        #     print("?'")
        # print(np.mean(data[dataMgr.regionBList[idx]]).shape)
        out[reg.regBoolList[idx]] = np.mean(data[reg.regBoolList[idx]],axis = 0)
    if drawEdge:
        edge = nodelib.getEdge(reg.IntMatrix)

        # out[edge] = [255,255,255]
        out[edge] = np.array([0,0,0])

    return out
      

def getSimpSegment(data, lamda = 0.2, num = 20, iteration=15):
    size = data.shape

    posList = []
    x_arr = np.array([np.array(range(size[0])),]*size[1]).T
    y_arr = np.array([np.array(range(size[1])),]*size[0])
    # print(size)
    
    # print(x_arr.shape)
    # print(y_arr.shape)
    
    r_arr = data[:,:,0]
    g_arr = data[:,:,1]
    b_arr = data[:,:,2]

    # decided init center
    for i in range(num):
        pos_x,pos_y = random.randint(0,size[0]-1),random.randint(0,size[1]-1)
        c_r,c_g,c_b = tuple(data[pos_x,pos_y])
        posList.append((pos_x,pos_y,c_r,c_g,c_b))
    
    dists = np.zeros((num,size[0],size[1]))
    blocks = np.zeros((num,size[0],size[1]), bool)

    recaculate = 0
    while True:
        # get dist
        for i in range(num):
            (pos_x,pos_y,c_r,c_g,c_b) = posList[i]
            x_dist = x_arr-pos_x
            y_dist = y_arr-pos_y
            r_dist = r_arr-c_r
            g_dist = g_arr-c_g
            b_dist = b_arr-c_b

            dists[i] = np.sqrt(np.square(x_dist) + np.square(y_dist) + \
                lamda*(np.square(r_dist) + np.square(g_dist) + np.square(b_dist)) )
        # get minimum 
        dmax = np.amin(dists, axis=0)
        # get region 
        for i in range(num):
            (pos_x,pos_y,c_r,c_g,c_b) = posList[i]
            blocks[i] = np.equal(dists[i], dmax)

        if recaculate == iteration-1:
            break
        recaculate += 1
        # get new center
        posList.clear()
        for i in range(num):
            block = blocks[i]
            # print(x_arr.shape)
            # print(block.shape)
            pos_x = np.mean(x_arr[block])
            pos_y = np.mean(y_arr[block])
            c_r = np.mean(r_arr[block])
            c_g = np.mean(g_arr[block])
            c_b = np.mean(b_arr[block])
            posList.append((pos_x,pos_y,c_r,c_g,c_b))
    
    region = np.zeros((size[0],size[1]))
    for i in range(num):
        region[blocks[i]] = i+1

    return region

def mergeRegionAdj(dataMgr, threshold):
    data = dataMgr.Cdata
    size = dataMgr.shape
    regions = dataMgr.region.IntMatrix

    # if len(size) == 3:
    #     print("randomseg.mergeRegionAdj: wrong input size!")
    #     return regions
    cy_d = data[:-1,:,0]-data[1:,:,0]
    cy_r = data[:,:-1,0]-data[:,1:,0]
    cb_d = data[:-1,:,1]-data[1:,:,1]
    cb_r = data[:,:-1,1]-data[:,1:,1]
    cr_d = data[:-1,:,2]-data[1:,:,2]
    cr_r = data[:,:-1,2]-data[:,1:,2]
    dif_d = np.square(cy_d)+np.square(cb_d)+np.square(cr_d)
    dif_r = np.square(cy_r)+np.square(cb_r)+np.square(cr_r)

    edge_d = regions[:-1,:] != regions[1:,:]
    edge_r = regions[:,:-1] != regions[:,1:]

    # print("d down:",np.mean(dif_d))
    # print("d right:",np.mean(dif_r))

    for x in range(size[0]-1):
        for y in range(size[1]):
            if edge_d[x,y] and (dif_d[x,y]<=threshold):
                dataMgr.region.mergeRegion((x+1,y),(x,y))#, visited)


    for x in range(size[0]):
        for y in range(size[1]-1):
            if edge_r[x,y] and (dif_r[x,y]<=threshold):
                dataMgr.region.mergeRegion((x,y+1),(x,y))#, visited)

# Belong to getDFSRegions
def getStartNode(region):
    size = region.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if region[i,j]:
                return (i,j)

def getRegDFS(region,node):
    x,y = node
    reg2 = np.zeros(region.shape,dtype=bool)
    value = region[x,y]
    # print(region.size)
    # region[x,y] = False
    reg2[x,y] = True
    l1=[node]
    del node
    size = region.shape
    while(len(l1) !=0):
        node1 = l1.pop()
        nx,ny = node1
        nList = nodelib.getNearNode(nx,ny,size[0],size[1])
        for node2 in nList:
            # mx,my = node2
            if not reg2[node2] and region[node2]==value:
                l1.append(node2)
                # region[mx,my] = False
                reg2[node2] = True

        del nList, node1, nx,ny

    return reg2

# def getDFSRegions(region):

#     r1 = np.copy(region)
#     l1 = []
#     while(np.max(r1) != 0):
#         node = getStartNode(r1)
#         reg_new = getRegDFS(r1,node)
#         l1.append(reg_new)
#         r1[reg_new] = False
#     return l1
                    

# def cutRegion(dataMgr):
#     cpRegionI = np.copy(dataMgr.regionInt)
#     cpSumList = dataMgr.regionSumList.copy()
#     cpBList = dataMgr.regionBList.copy()

#     for i in range(dataMgr.regionSize):
#         idx = i+1
#         if dataMgr.regionSumList[idx]<=0:
#             continue
#         regList = getDFSRegions(cpBList[idx])
#         dataMgr.regionBList[idx] = regList[0]
#         dataMgr.regionSumList[idx] = np.sum(regList[0])
#         for j in range(1,len(regList)):
#             dataMgr.regionBList.append(regList[j])
#             dataMgr.regionSumList.append(np.sum(regList[j]))
#             dataMgr.regionNum += 1
#             dataMgr.regionSize += 1
#             dataMgr.regionInt[regList[j]] = dataMgr.regionSize-1


def getChara2(data, ws):
    w1,w2,w3,w4 = ws
    cy_d = data[:-1,:,0]-data[1:,:,0]
    cy_r = data[:,:-1,0]-data[:,1:,0]
    cb_d = data[:-1,:,1]-data[1:,:,1]
    cb_r = data[:,:-1,1]-data[:,1:,1]
    cr_d = data[:-1,:,2]-data[1:,:,2]
    cr_r = data[:,:-1,2]-data[:,1:,2]
    cdif_d = w1*np.abs(cy_d)+w2*np.abs(cb_d)+w2*np.abs(cr_d)
    cdif_r = w1*np.abs(cy_r)+w2*np.abs(cb_r)+w2*np.abs(cr_r)
    covEdge = seglib.getEdge(data)
    covRidge = seglib.getRidge(data)
    diff_d = cdif_d + w3*covEdge[:-1] + w4*covRidge[:-1]
    diff_r = cdif_r + w3*covEdge[:,:-1] + w4*covRidge[:,:-1]
    return diff_d,diff_r

def mergeRegion(dataMgr,chara_d,chara_r, threshold):
    data = dataMgr.Cdata
    size = dataMgr.shape
    regions = dataMgr.region.IntMatrix
    difReg_d = regions[:-1,:] != regions[1:,:]
    difReg_r = regions[:,:-1] != regions[:,1:]
    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] :
                if chara_d[x,y]<=threshold:
                    dataMgr.region.mergeRegion((x+1,y),(x,y))#, visited)


    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y]:
                if chara_r[x,y]<=threshold:
                    dataMgr.region.mergeRegion((x,y+1),(x,y))#, visited)

def areaChara(area,t1,t2):
    if area<t1:
        return 1
    if area>t2:
        return 0
    return (area-t1)/(t2-t1)

def mergeRegionContainArea(dataMgr,ws,ts, threshold):

    t1,t2 = ts

    data = dataMgr.Cdata
    size = dataMgr.shape
    chara_d,chara_r = getChara2(data,ws)
    regions = dataMgr.region.IntMatrix
    regMgr = dataMgr.region
    difReg_d = regions[:-1,:] != regions[1:,:]
    difReg_r = regions[:,:-1] != regions[:,1:]
    l_a = threshold

    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] :
                i1,i2 = dataMgr.getNodeIdx((x+1,y)),dataMgr.getNodeIdx((x,y))
                area = min(regMgr.regSumList[i1],regMgr.regSumList[i2])
                if chara_d[x,y]-l_a*areaChara(area,t1,t2)<=threshold:
                    regMgr.mergeRegion((x+1,y),(x,y))#, visited)


    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y]:
                i1,i2 = dataMgr.getNodeIdx((x,y+1)),dataMgr.getNodeIdx((x,y))
                area = min(regMgr.regSumList[i1],regMgr.regSumList[i2])
                if chara_r[x,y]-l_a*areaChara(area,t1,t2)<=threshold:
                    regMgr.mergeRegion((x,y+1),(x,y))#, visited)
def mergeRegionBy2(dataMgr,ws, threshold):

    
    data = dataMgr.Cdata
    # th_sq = threshold**2

    # if len(size) == 3:
    #     print("randomseg.mergeRegionAdj: wrong input size!")
    #     return regions
    diff_d,diff_r = getChara2(data,ws)
    mergeRegion(dataMgr,diff_d,diff_r,threshold)


def meanmergeRegionAdj(dataMgr, threshold):
    
    data = dataMgr.Cdata
    cRegion = toMean(dataMgr,False)
    # print(cRegion.dtype)
    # sys.exit(0)
    size = dataMgr.shape
    regions = dataMgr.region.IntMatrix
    # th_sq = threshold**2
    
    # size = data.shape

    visited = np.zeros((size[0],size[1]))
    # if len(size) == 3:
    #     print("randomseg.mergeRegionAdj: wrong input size!")
    #     return regions
    cy_d = cRegion[:-1,:,0]-cRegion[1:,:,0]
    cy_r = cRegion[:,:-1,0]-cRegion[:,1:,0]
    cb_d = cRegion[:-1,:,1]-cRegion[1:,:,1]
    cb_r = cRegion[:,:-1,1]-cRegion[:,1:,1]
    cr_d = cRegion[:-1,:,2]-cRegion[1:,:,2]
    cr_r = cRegion[:,:-1,2]-cRegion[:,1:,2]
    dif_d = np.square(cy_d)+np.square(cb_d)+np.square(cr_d)
    dif_r = np.square(cy_r)+np.square(cb_r)+np.square(cr_r)

    difReg_d = regions[:-1,:] != regions[1:,:]
    difReg_r = regions[:,:-1] != regions[:,1:]

    # print("dmin down:",np.min(dif_d))
    # print("dmin right:",np.min(dif_r))
    # print("d down:",np.mean(dif_d))
    # print("d right:",np.mean(dif_r))
    # print("dtype:",dif_r.dtype)

    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] and (dif_d[x,y]<=threshold):
            # if difReg_d[x,y] and nodelib.arr_equal(cRegion[x,y],cRegion[x+1,y]):
                # print(cRegion[x,y],cRegion[x+1,y])
                dataMgr.region.mergeRegion((x+1,y),(x,y))#, visited)


    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y] and (dif_r[x,y]<=threshold):
            # if difReg_r[x,y] and nodelib.arr_equal(cRegion[x,y],cRegion[x,y+1]):
                # print(cRegion[x,y],cRegion[x,y+1])
                dataMgr.region.mergeRegion((x,y+1),(x,y))#, visited)

    # imglib.showImg(imglib.arrToImg(visited))
    # print(np.mean(visited))
    # print(np.mean(visited[visited != 0]))

def mergeLittleRegion(dataMgr,ws, threhold):
    regMgr = dataMgr.region
    chara0 = toMean(dataMgr)
    chara = np.sum(abs(chara0[:-1]-chara0[1:]),axis=2),np.sum(abs(chara0[:,:-1]-chara0[:,1:]),axis=2)
    size = dataMgr.shape
    for i in range(regMgr.regSize ):
        idx = i+1
        if regMgr.regSumList[idx] == 0:
            continue
        if regMgr.regSumList[idx] <= threhold:
            reg1 = regMgr.regBoolList[idx]
            edge = nodelib.getInnerFrame(reg1)
            dif_min = 100000
            new_reg_idx = 0
            for x in range(size[0]):
                for y in range(size[1]):
                    if edge[x,y]:
                        # nlist= nodelib.getNearNode(x,y,size[0],size[1])
                        if x!=0 and not reg1[x-1,y] and chara[0][x-1,y]<dif_min:
                            new_reg_idx = regMgr.IntMatrix[x-1,y]
                            dif_min = chara[0][x-1,y]
                        if y!=0 and not reg1[x,y-1] and chara[1][x,y-1]<dif_min:
                            new_reg_idx = regMgr.IntMatrix[x,y-1]
                            dif_min = chara[1][x,y-1]
                        if x!=size[0]-1 and not reg1[x+1,y] and chara[0][x,y]<dif_min:
                            new_reg_idx = regMgr.IntMatrix[x+1,y]
                            dif_min = chara[0][x,y]
                        if y!=size[1]-1 and not reg1[x,y+1] and chara[1][x,y]<dif_min:
                            new_reg_idx = regMgr.IntMatrix[x,y+1]
                            dif_min = chara[1][x,y]
            
            if new_reg_idx==0:
                print("? ? ?")
                return
            regMgr.mergeRegion(idx,new_reg_idx)



                

def getLargeSegment(dataMgr, num = 1000, l1=2,l2=1, move_step = 4):
    '''
    l1: cbcr gain
    l2: dist gain
    '''

    size = dataMgr.shape
    region = dataMgr.region.IntMatrix.copy()
    # region = dataMgr.shape

    # size = data.shape
    # region = np.zeros((size[0],size[1]))

    dist_max = l2*(size[0]+size[1])+255+l1*(255*2)
    dist = np.ones((size[0],size[1]))*dist_max

    cut_num = 8

    cut_num_x, cut_num_y = cut_num, cut_num
    cut_size_x = size[0]/cut_num
    cut_size_y = size[1]/cut_num
    if cut_size_x>cut_size_y:
        while size[0]/(cut_num_x+1) > cut_size_y:
            cut_num_x += 1
        cut_size_x = size[0]/cut_num_x
    if cut_size_x<cut_size_y:
        while size[1]/(cut_num_y+1) > cut_size_x:
            cut_num_y += 1
        cut_size_y = size[1]/cut_num_y
    
    def getRange(x,y):
        if x<0 or x>=size[0]:
            return None
        if y<0 or y>=size[1]:
            return None
        m = int(x/(cut_size_x))
        if m==0:
            x1,x2 = 0, int(cut_size_x*1.5)
        elif m==cut_num_x-1:
            x1,x2 = size[0]-int(cut_size_x*1.5), size[0]
        else:
            x1,x2 = int(cut_size_x*(m-0.5)), int(cut_size_x*(m+1.5))
        
        n = int(y/(cut_size_y))
        if n==0:
            y1,y2 = 0, int(cut_size_y*1.5)
        elif n==cut_num_y-1:
            y1,y2 = size[1]-int(cut_size_y*1.5), size[1]
        else:
            y1,y2 = int(cut_size_y*(n-0.5)), int(cut_size_y*(n+1.5))

        return x1,x2,y1,y2



    posList = []
    x_arr = np.array([np.array(range(size[0])),]*size[1]).T
    y_arr = np.array([np.array(range(size[1])),]*size[0])
    # print(size)
    
    # print(x_arr.shape)
    # print(y_arr.shape)
    
    AdjData = dataMgr.Cdata
    cy_arr = AdjData[:,:,0]
    cb_arr = AdjData[:,:,1]
    cr_arr = AdjData[:,:,2]
    grad_func = cy_arr + np.sqrt(l1)*(cb_arr+cr_arr)
    # dataMgr.grad = nodelib.getGradient(grad_func)

    # decided init center
    for i in range(num):
        ox,oy = random.randint(0,size[0]-1),random.randint(0,size[1]-1)
        ppos = (ox,oy)
        for j in range(move_step):
            # tx,ty = 
            minGrad = dataMgr.grad[ppos[0],ppos[1]]
            npos = ppos
            # (ox,oy)
            dir = Direction()
            for dirIdx in range(8):
                tem_x,tem_y = dir.getEndPos(ppos)
                if tem_x<0 or tem_x>=size[0]:
                    continue
                if tem_y<0 or tem_y>=size[1]:
                    continue
                
                if dataMgr.grad[tem_x,tem_y] < minGrad:
                    npos = (tem_x,tem_y)
                    minGrad = dataMgr.grad[tem_x,tem_y]
                dir.rotate()
            if npos==ppos:
                break
        
        pos_x,pos_y = npos



        cy,cb,cr = tuple(AdjData[pos_x,pos_y])
        posList.append((pos_x,pos_y,cy,cb,cr))
    
    sortedList = sorted(posList)
    # for i in range(num):

    # dists = np.zeros((num,size[0],size[1]))
    # blocks = np.zeros((num,size[0],size[1]), bool)
    prepos = (-1,-1)
    # dataMgr.regionSumList = [-1]
    # dataMgr.regionSize = num
    # 
    for i in range(num):
        # dataMgr.regionSumList.append(0)
        # get dist
        (pos_x,pos_y,c_y,c_b,c_r) = sortedList[i]
        if (pos_x,pos_y) == prepos:
            # dataMgr.regionBList = 0
            continue
        # dataMgr.regionNum += 1
        prepos = (pos_x,pos_y)
        r_x1, r_x2, r_y1, r_y2 = getRange(pos_x,pos_y)
        x_dist = x_arr[r_x1:r_x2, r_y1:r_y2]-pos_x
        y_dist = y_arr[r_x1:r_x2, r_y1:r_y2]-pos_y
        cy_dist = cy_arr[r_x1:r_x2, r_y1:r_y2]-c_y
        cb_dist = cb_arr[r_x1:r_x2, r_y1:r_y2]-c_b
        cr_dist = cr_arr[r_x1:r_x2, r_y1:r_y2]-c_r

        dist_tem = np.sqrt( l2*(np.square(x_dist) + np.square(y_dist)) + \
            np.square(cy_dist) + l1*(np.square(cb_dist) + np.square(cr_dist)) )
        
        # b_less = np.less(dist_tem,dist[r_x1:r_x2, r_y1:r_y2])
        # print(str(len(dataMgr.regionSumList))+ ':')
        for x in range(r_x1,r_x2):
            for y in range(r_y1, r_y2):
                # print(r_x1,r_x2,r_y1, r_y2)
                if dist_tem[x-r_x1,y-r_y1] < dist[x,y]:
                    # not region[x,y] and 
                    old_reg = region[x,y]
                    dist[x,y] = dist_tem[x-r_x1,y-r_y1]
                    region[x,y] = i+1
                    # print(i+1)
                    # dataMgr.regionSumList[old_reg] -= 1
                    # dataMgr.regionSumList[i+1] += 1
    # dataMgr.regionBList = [0]

    region_visited = np.zeros(size[0:2], dtype = bool)
    # print(region_visited.shape)
    for i in range(size[0]):
        for j in range(size[1]):
            if region_visited[i,j]:
                continue
            reg1 = getRegDFS(region, (i,j))
            region_visited[reg1] = True
            # idx = i+1
            dataMgr.region.addRegion(reg1) #np.zeros((size[0],size[1]),bool)

            del reg1

        


    # dataMgr.settleRegion()
    return region

l1 = 0
l2 = 0
w1 = 0
w2 = w1*l1
w3 = 0
w4 = 0
threshold = 0
t1 = 0
t2 = 1
def setPara(gray, shape):

    if not DEF_LDATA:
        l1 = 2
        l2 = 0.5
        w1 = 1
        w2 = w1*l1
        w3 = 0.8
        w4 = 1
        threshold = 25
        t1 = 64
        t2 = int(max(400,max(shape[0],shape[1])/10.0*4,shape[0]*shape[1]/1600))
        # t2 = 400
    if DEF_LDATA:
        if not gray:
            l1 = 4
            l2 = 0.5
            w1 = 1
            w2 = w1*l1
            w3 = 0.8
            w4 = 1
            threshold = 15
            t1 = 64
            t2 = int(max(400,max(shape[0],shape[1])/10.0*4,shape[0]*shape[1]/1600))
        else:
            print("gray")
            l1 = 0
            l2 = 0.2
            w1 = 1
            w2 = w1*l1
            w3 = 0.8
            w4 = 1
            threshold = 10
            t1 = 64
            t2 = int(max(400,max(shape[0],shape[1])/10.0*4,shape[0]*shape[1]/1600))

def processFile(data):
    dataMgr = DataMgr(data)
    
    setPara(imglib.isGray(data), dataMgr.shape )
    
    start_time = time.time()
    getLargeSegment(dataMgr, 1000,l1,l2)
    output0 = dataMgr.region_copy()
    if np.mean(output0)!=np.mean(dataMgr.region.IntMatrix):
        print("Wrong!")
    print("RandomSeg edge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    oc = []
    oce = []
    oc.append(dataMgr.data)
    oce.append(np.zeros(dataMgr.data.shape))
    # oc0 = toMean(dataMgr,False)
    # oce0 = toMean(dataMgr,True)
    # out1 = dataMgr.region_copy()
    mergeThreshold = 2
    meanmergeRegionAdj(dataMgr, mergeThreshold)
    # mergeThreshold = 0
    # mergeRegionAdj(dataMgr, mergeThreshold)
    print("Mean merge while lamda=%d time:\t--- %8.4f seconds ---" % (mergeThreshold,time.time() - start_time))
    
    oc.append(toMean(dataMgr,False))
    oce.append(toMean(dataMgr,True))

    # mergeThreshold = 0

    start_time = time.time()
    mergeRegionBy2(dataMgr, (w1,w2,w3,w4),threshold)
    # out2 = dataMgr.region_copy()
    print("Original merge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
    oc.append(toMean(dataMgr,False))
    oce.append(toMean(dataMgr,True))

    # threshold = 16
    start_time = time.time()
    mergeLittleRegion(dataMgr, (w1,w2,w3,w4),t1)
    # out2 = dataMgr.region_copy()
    print("Little region merge time:\t--- %8.4f seconds ---" % (time.time() - start_time))
    oc.append(toMean(dataMgr,False))
    oce.append(toMean(dataMgr,True))

    # print(dataMgr.regionSize)
    # print(len(dataMgr.regionBList))
    # print(len(dataMgr.regionSumList))

    start_time = time.time()
    mergeRegionContainArea(dataMgr, (w1,w2,w3,w4),(t1,t2),threshold)
    # out2 = dataMgr.region_copy()
    print("Area merge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
    oc.append(toMean(dataMgr,False))
    oce.append(toMean(dataMgr,True))

    outRow1 = imglib.mergeArray(tuple(oc),axis=1, interval=20)
    outRow2 = imglib.mergeArray(tuple(oce),axis=1, interval=20)
    # out2 = imglib.mergeArray(tuple(list2),axis=1, interval=20)

    # out = imglib.mergeArray((out1,out2),0,20)
    return imglib.mergeArray((outRow1,outRow2),axis=0, interval=20)


def testFile(data):
    dataMgr = DataMgr(data)

    t2 = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.shape[0]*dataMgr.shape[1]/1600))

    
    start_time = time.time()
    getLargeSegment(dataMgr, 1000,l1,l2)
    output0 = dataMgr.region_copy()
    if np.mean(output0)!=np.mean(dataMgr.region.IntMatrix):
        print("Wrong!")
    print("RandomSeg edge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
    
    start_time = time.time()
    oc = []
    oce = []
    oc.append(dataMgr.data)
    oce.append(np.zeros(dataMgr.data.shape))
    oc0 = toMean(dataMgr,False)
    oce0 = toMean(dataMgr,True)
    # out1 = dataMgr.region_copy()
    mergeThreshold = 2
    meanmergeRegionAdj(dataMgr, mergeThreshold)
    # mergeThreshold = 0
    # mergeRegionAdj(dataMgr, mergeThreshold)
    print("Mean merge while lamda=%d time:\t--- %8.4f seconds ---" % (mergeThreshold,time.time() - start_time))
    
    oc.append(toMean(dataMgr,False))
    oce.append(toMean(dataMgr,True))

    outRow1 = imglib.mergeArray(tuple(oc),axis=1, interval=20)
    outRow2 = imglib.mergeArray(tuple(oce),axis=1, interval=20)
    # out2 = imglib.mergeArray(tuple(list2),axis=1, interval=20)

    # out = imglib.mergeArray((out1,out2),0,20)
    return imglib.mergeArray((outRow1,outRow2),axis=0, interval=20)
    


