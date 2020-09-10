import numpy as np
import random 
# import sys
# sys.path.append(".")
from util import nodelib, imglib
import time

in_path_str = "Pic/"

DEBUG = False

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

def getLargeSegment(data, num = 500, l1=2,l2=1, move_step = 4):


    size = data.shape
    region = np.zeros((size[0],size[1]))

    dist_max = l2*(size[0]+size[1])+255+l1*(255*2)
    dist = np.ones((size[0],size[1]))*dist_max

    cut_num = 6
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
    
    YCbCr_data = imglib.RGBtoYCbCr(data)
    cy_arr = YCbCr_data[:,:,0]
    cb_arr = YCbCr_data[:,:,1]
    cr_arr = YCbCr_data[:,:,2]
    grad_func = cy_arr + np.sqrt(l1)*(cb_arr+cr_arr)
    grad = nodelib.getGradient(grad_func)

    # decided init center
    for i in range(num):
        ox,oy = random.randint(0,size[0]-1),random.randint(0,size[1]-1)
        ppos = (ox,oy)
        for j in range(move_step):
            # tx,ty = 
            minGrad = grad[ppos[0],ppos[1]]
            npos = ppos
            # (ox,oy)
            dir = Direction()
            for dirIdx in range(8):
                tem_x,tem_y = dir.getEndPos(ppos)
                if tem_x<0 or tem_x>=size[0]:
                    continue
                if tem_y<0 or tem_y>=size[1]:
                    continue
                
                if grad[tem_x,tem_y] < minGrad:
                    npos = (tem_x,tem_y)
                    minGrad = grad[tem_x,tem_y]
                dir.rotate()
            if npos==ppos:
                break
        
        pos_x,pos_y = npos



        cy,cb,cr = tuple(YCbCr_data[pos_x,pos_y])
        posList.append((pos_x,pos_y,cy,cb,cr))
    
    # dists = np.zeros((num,size[0],size[1]))
    # blocks = np.zeros((num,size[0],size[1]), bool)

   
    
    for i in range(num):
        # get dist
        (pos_x,pos_y,c_y,c_b,c_r) = posList[i]
        r_x1, r_x2, r_y1, r_y2 = getRange(pos_x,pos_y)
        x_dist = x_arr[r_x1:r_x2, r_y1:r_y2]-pos_x
        y_dist = y_arr[r_x1:r_x2, r_y1:r_y2]-pos_y
        cy_dist = cy_arr[r_x1:r_x2, r_y1:r_y2]-c_y
        cb_dist = cb_arr[r_x1:r_x2, r_y1:r_y2]-c_b
        cr_dist = cr_arr[r_x1:r_x2, r_y1:r_y2]-c_r

        dist_tem = np.sqrt( l2*(np.square(x_dist) + np.square(y_dist)) + \
            np.square(cy_dist) + l1*(np.square(cb_dist) + np.square(cr_dist)) )
        
        # b_less = np.less(dist_tem,dist[r_x1:r_x2, r_y1:r_y2])
        for x in range(r_x1,r_x2):
            for y in range(r_y1, r_y2):
                # print(r_x1,r_x2,r_y1, r_y2)
                if dist_tem[x-r_x1,y-r_y1] < dist[x,y]:
                    dist[x,y] = dist_tem[x-r_x1,y-r_y1]
                    region[x,y] = i+1
        


    
  
    
    # region = np.zeros((size[0],size[1]))
    # for i in range(num):
    #     region[blocks[i]] = i+1

    return region

def processFile(data):
    array0 = data
    list0 = []
    list0_o = []
    for i in range(8):

        seg = getLargeSegment(array0, 500,2,0.5)
        # print("Dealing time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        list0.append(seg)
        list0_o.append(nodelib.toColor(seg))
        # start_time = time.time()
        # seg_color = nodelib.toColor(seg)
        # list0.append(seg_color)
    
    
    blend1 = 255-nodelib.combineEdge(list0)
    list1 = list0_o[0:4]
    list1.insert(0,array0)
    list2 = list0_o[4:8]
    list2.insert(0,blend1)
    # out2 = 255*(1-out1>int(255/8*4-10))

    out1 = imglib.mergeArray(tuple(list1),axis=1, interval=20)
    out2 = imglib.mergeArray(tuple(list2),axis=1, interval=20)

    out = imglib.mergeArray((out1,out2),0,20)
    return out
    # imglib.saveImg(out,out_path)
    # print(out2.shape)
    
    # return out1,out2
    # return out1

'''
def processFile_backup(data, lamda):
    array0 = data

    # print("Shape: ", end="")
    # print(array0.shape)
    # array1 = imglib.img3dTo2d(array0)

    list0 = []
    list0_o = []
    start_time = time.time()
    
    for i in range(6):

        seg = getSegment(array0,lamda=lamda, num=20, iteration=10)
        # print("Dealing time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        list0_o.append(seg)
        # start_time = time.time()
        # seg_color = nodelib.toColor(seg)
        # list0.append(seg_color)
    
    # list1 = list0[0:4]
    # list1.insert(0,array0)
    # list2 = list0[4:8]
    # list2.insert(0,255-nodelib.combineEdge(list0_o))
    out1 = nodelib.combineEdge(list0_o)
    out2 = out1>int(255/6*3-10)
    # print(out2.shape)

    # out1 = imglib.mergeArray(tuple(list1),axis=1, interval=20)
    # out2 = imglib.mergeArray(tuple(list2),axis=1, interval=20)

    # output = imglib.mergeArray((array0, 255-out1, 255*(1-out2)),axis=1, interval=20)
    

    return out1,out2

    
    # print("Dealing time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))


# '''
        


        





