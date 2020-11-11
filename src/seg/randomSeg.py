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
        



def toMean(dataMgr, drawEdge = False, cEdge =  False, alert = True):
    size_x, size_y = dataMgr.shape
    regMgr = dataMgr.regMgr

    data = dataMgr.data_copy()
    out = np.zeros((size_x, size_y, 3))
    count = 0
    region_visited = np.zeros((size_x, size_y),dtype=bool)
    for i in range( size_x):
        for j in range(size_y):
            if region_visited[i,j]:
                continue
            # idx = regMgr.space[i,j]
            reg = regMgr.getRegion((i,j))
            datalist = []
            for pos in reg:
                datalist.append(data[pos])
            regmean = np.mean(np.array(datalist),axis=0)
            for pos in reg:
                out[pos] = regmean
                region_visited[pos] = True
                
            # if np.sum(reg)==0:
            #     continue
            count += 1
    if count > regMgr.idxNum:
        print("?count out: (",count, ", ",regMgr.idxNum ,").")
        # if np.sum(dataMgr.regionBList[idx]) == 0:
        #     print("?'")
    if alert:
        reg0 = regMgr.space==0
        out[reg0] = np.array([255,0,0])
    if drawEdge:
        edge = nodelib.getEdge(regMgr.space)

        # out[edge] = [255,255,255]
        if cEdge:
            out[edge] = np.array([255,106,52])
        else:
            out[edge] = np.array([8,8,8])

    return out
      


# def getDFSRegions(region):

#     r1 = np.copy(region)
#     p_seg_color = []
#     while(np.sum(r1) != 0):
#         node = getStartNode(r1)
#         reg_new = getRegDFS(r1,node)
#         p_seg_color.append(reg_new)
#         r1[reg_new] = False
#     return p_seg_color


# def testSimpleReg(region):
#     r1 = np.copy(region)
#     node = getStartNode(r1)
#     reg_new = getRegDFS(r1,node)
#     if np.sum(reg_new) < np.sum(r1):
#         return False
#     return True             

# def cutScatterRegion(dataMgr, threhold=1):
#     # region = dataMgr.region.space
#     regMgr = dataMgr.regMgr
#     space_copy = dataMgr.region_copy()


#     for i in range(regMgr.idxMax):
#         idx = i+1
#         reg0 = space_copy==idx
#         # print(reg0.shape)
#         if np.sum(reg0)==0:
#             continue
#         regList = getDFSRegions(reg0)
#         if len(regList)==1:
#             if np.sum(reg0) <= threhold:
#                 regMgr.labelLittleReg(reg0)
#             continue
#         # reg.space[BList[idx]] = 0
#         # reg.space[regList[0]] = idx
#         # BList[idx] = regList[0]
#         # sumList[idx] = np.sum(regList[0])
#         # reg.mergeLittleReg(idx,threhold)

#         for reg_i in range(1,len(regList)):
#             if not regMgr.cutRegion(idx,regList[reg_i]):
#                 print("wee?")
#             if np.sum(regList[reg_i]) <= threhold:
#                 regMgr.labelLittleReg(regList[reg_i])

#     print("Finished cut")

#     chara = getChara2(dataMgr.Cdata, dataMgr.para)

#         # if not testSimpleReg(BList[idx]):
#         #     print("tReg: cut fail.")
#     regMgr.mergeLabelReg(chara,threhold)

# Belong to getDFSRegions 
# ----------------------------------------------------- #


# The overwhelming simple merge way
def mergeRegionAdj(dataMgr, threshold):
    data = dataMgr.Cdata
    size = dataMgr.shape
    regions = dataMgr.regMgr.space

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


    for x in range(size[0]-1):
        for y in range(size[1]):
            if edge_d[x,y] and (dif_d[x,y]<=threshold):
                dataMgr.regMgr.mergeRegion((x+1,y),(x,y))#, visited)


    for x in range(size[0]):
        for y in range(size[1]-1):
            if edge_r[x,y] and (dif_r[x,y]<=threshold):
                dataMgr.regMgr.mergeRegion((x,y+1),(x,y))#, visited)


def mergeRegion(dataMgr,chara_d,chara_r):
    threshold = dataMgr.para.p_cha_thre
    data = dataMgr.Cdata
    size = dataMgr.shape
    regions = dataMgr.regMgr.space
    difReg_d = regions[:-1,:] != regions[1:,:]
    difReg_r = regions[:,:-1] != regions[:,1:]
    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] :
                if chara_d[x,y]<=threshold:
                    dataMgr.regMgr.mergeRegion((x+1,y),(x,y))


    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y]:
                if chara_r[x,y]<=threshold:
                    dataMgr.regMgr.mergeRegion((x,y+1),(x,y))


def meanmergeRegionAdj(dataMgr, threshold):
    
    data = dataMgr.Cdata
    cRegion = toMean(dataMgr,False)

    size = dataMgr.shape
    regions = dataMgr.regMgr.space
    visited = np.zeros((size[0],size[1]))
    
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


    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] and (dif_d[x,y]<=threshold):
                dataMgr.regMgr.mergeRegion((x+1,y),(x,y))


    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y] and (dif_r[x,y]<=threshold):
                dataMgr.regMgr.mergeRegion((x,y+1),(x,y))


def getChara2(data,para):
    wc1 = para.p_cha_wc1
    wc23 = para.p_cha_wc23
    we = para.p_cha_we
    wr = para.p_cha_wr

    cy_d = data[:-1,:,0]-data[1:,:,0]
    cy_r = data[:,:-1,0]-data[:,1:,0]
    cb_d = data[:-1,:,1]-data[1:,:,1]
    cb_r = data[:,:-1,1]-data[:,1:,1]
    cr_d = data[:-1,:,2]-data[1:,:,2]
    cr_r = data[:,:-1,2]-data[:,1:,2]
    cdif_d = wc1*np.abs(cy_d)+wc23*np.abs(cb_d)+wc23*np.abs(cr_d)
    cdif_r = wc1*np.abs(cy_r)+wc23*np.abs(cb_r)+wc23*np.abs(cr_r)

    covEdge = seglib.getEdge(data)
    covRidge = seglib.getRidge(data)
    edge_d = (covEdge[0:-1]+covEdge[1:])/2.0
    edge_r = (covEdge[:,0:-1]+covEdge[:,1:])/2.0
    ridge_d = (covRidge[0:-1]+covRidge[1:])/2.0
    ridge_r = (covRidge[:,0:-1]+covRidge[:,1:])/2.0

    diff_d = cdif_d + we*covEdge[:-1] + wr*covRidge[:-1]
    diff_r = cdif_r + we*covEdge[:,:-1] + wr*covRidge[:,:-1]
    return diff_d,diff_r

def getChara3(dataMgr):
    # dataMgr.setMeanGradArr()
    
    wc1 = dataMgr.para.p_cha_wc1
    wc23 = dataMgr.para.p_cha_wc23
    we = dataMgr.para.p_cha_we
    wr = dataMgr.para.p_cha_wr
    alpha = dataMgr.para.p_gd_pow
    g_we = dataMgr.para.p_gd_we
    g_wr = dataMgr.para.p_gd_wr
    data = dataMgr.Cdata

    cy_d = data[:-1,:,0]-data[1:,:,0]
    cy_r = data[:,:-1,0]-data[:,1:,0]
    cb_d = data[:-1,:,1]-data[1:,:,1]
    cb_r = data[:,:-1,1]-data[:,1:,1]
    cr_d = data[:-1,:,2]-data[1:,:,2]
    cr_r = data[:,:-1,2]-data[:,1:,2]
    cdif_d = wc1*np.abs(cy_d)+wc23*np.abs(cb_d)+wc23*np.abs(cr_d)
    cdif_r = wc1*np.abs(cy_r)+wc23*np.abs(cb_r)+wc23*np.abs(cr_r)

    grad = dataMgr.getMeanGrad()
    grad_d = (grad[0:-1]+grad[1:])/2.0
    grad_r = (grad[:,0:-1]+grad[:,1:])/2.0
    grad_d[grad_d==0] = 1E-6
    grad_r[grad_r==0] = 1E-6

    covEdge = seglib.getEdge(data)
    covRidge = seglib.getRidge(data)
    edge_d = (covEdge[0:-1]+covEdge[1:])/2.0
    edge_r = (covEdge[:,0:-1]+covEdge[:,1:])/2.0
    ridge_d = (covRidge[0:-1]+covRidge[1:])/2.0
    ridge_r = (covRidge[:,0:-1]+covRidge[:,1:])/2.0

    diff_d = cdif_d + (we*g_we*edge_d + wr*g_wr*ridge_d) / np.power(grad_d, alpha)
    diff_r = cdif_r + (we*g_we*edge_r + wr*g_wr*ridge_r) / np.power(grad_r, alpha)
    return diff_d,diff_r

def mergeRegionBy2(dataMgr):

    
    data = dataMgr.Cdata
    
    
    diff_d,diff_r = getChara2(data,dataMgr.para)
    mergeRegion(dataMgr,diff_d,diff_r)

def mergeRegion_A_2(dataMgr):

    def areaChara(area,p_la_bottom,p_la_top):
        if area<p_la_bottom:
            return 1
        if area>p_la_top:
            return 0
        return (area-p_la_bottom)/(p_la_top-p_la_bottom)


    p_la_bottom = dataMgr.para.p_la_bottom
    p_la_top = dataMgr.para.p_la_top
    p_cha_thre = dataMgr.para.p_cha_thre

    data = dataMgr.Cdata
    size = dataMgr.shape
    chara_d,chara_r = getChara2(data,dataMgr.para)
    regions = dataMgr.regMgr.space
    regMgr = dataMgr.regMgr
    difReg_d = regions[:-1,:] != regions[1:,:]
    difReg_r = regions[:,:-1] != regions[:,1:]
    l_a = p_cha_thre

    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] :
                # i1,i2 = dataMgr.getNodeIdx((x+1,y)),dataMgr.getNodeIdx((x,y))
                area = min(regMgr.getRegArea((x+1,y)),regMgr.getRegArea((x,y)))
                if area==0:
                    print("randomSeg.mergeRegion_A_2: area=0")

                if chara_d[x,y]-l_a*areaChara(area,p_la_bottom,p_la_top)<=p_cha_thre:
                    regMgr.mergeRegion((x+1,y),(x,y))

    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y]:
                area = min(regMgr.getRegArea((x,y+1)),regMgr.getRegArea((x,y)))
                if chara_r[x,y]-l_a*areaChara(area,p_la_bottom,p_la_top)<=p_cha_thre:
                    regMgr.mergeRegion((x,y+1),(x,y))
    regMgr.settleRegion()
    # TODO
                    

def mergeRegion_AG_3(dataMgr):
    '''
    By area and region param
    Using chara3 (which consider meanGrad)
    '''

    def areaChara(area,p_la_bottom,p_la_top):
        if area<p_la_bottom:
            return 1
        if area>p_la_top:
            return 0
        return (area-p_la_bottom)/(p_la_top-p_la_bottom)


    p_la_bottom = dataMgr.para.p_la_bottom
    p_la_top = dataMgr.para.p_la_top
    p_gd_thre = dataMgr.para.p_gd_thre

    data = dataMgr.Cdata
    size = dataMgr.shape
    chara_d,chara_r = getChara3(dataMgr)
    regions = dataMgr.regMgr.space
    regMgr = dataMgr.regMgr
    difReg_d = regions[:-1,:] != regions[1:,:]
    difReg_r = regions[:,:-1] != regions[:,1:]
    l_a = p_gd_thre

    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] :
                area = min(regMgr.getRegArea((x+1,y)),regMgr.getRegArea((x,y)))
                if area==0:
                    print("randomSeg.mergeRegion_A_2: area=0")

                if chara_d[x,y]-l_a*areaChara(area,p_la_bottom,p_la_top)<=p_gd_thre:
                    regMgr.mergeRegion((x+1,y),(x,y))

    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y]:
                area = min(regMgr.getRegArea((x,y+1)),regMgr.getRegArea((x,y)))
                if area==0:
                    print("randomSeg.mergeRegion_A_2: area=0")
                if chara_r[x,y]-l_a*areaChara(area,p_la_bottom,p_la_top)<=p_gd_thre:
                    regMgr.mergeRegion((x,y+1),(x,y))

    regMgr.settleRegion()
                    

def mergeRegion_AG_31(dataMgr):
    '''
    By area and region param
    Using chara3 (which consider meanGrad)
    '''

    def areaChara(area,p_la_bottom,p_la_top):
        if area<p_la_bottom:
            return 1
        if area>p_la_top:
            return 0
        return (area-p_la_bottom)/(p_la_top-p_la_bottom)

    dataMgr.para.p_cha_we *= dataMgr.para.p_wth_amp
    dataMgr.para.p_cha_wr *= dataMgr.para.p_wth_amp
    p_la_bottom = dataMgr.para.p_la_bottom
    p_la_top = dataMgr.para.p_la_top
    p_gd_thre = dataMgr.para.p_wth_thre  
    p_aw = dataMgr.para.p_wth_area   
    p_gw = dataMgr.para.p_wth_grad    

    data = dataMgr.Cdata
    size = dataMgr.shape
    chara_d,chara_r = getChara3(dataMgr)
    regions = dataMgr.regMgr.space
    regMgr = dataMgr.regMgr
    difReg_d = regions[:-1,:] != regions[1:,:]
    difReg_r = regions[:,:-1] != regions[:,1:]
    l_a = p_gd_thre

    mg_d = np.minimum(dataMgr.grad[:-1],dataMgr.grad[1:])
    mg_d[mg_d==0] = 0.0001
    mg_r = np.minimum(dataMgr.grad[:,:-1],dataMgr.grad[:,1:])
    mg_r[mg_r==0] = 0.0001

    a_bottom = dataMgr.shape[0]*dataMgr.shape[1]/144.0

    # np.ones(dataMgr.shape)*
    th_d = p_gd_thre / (1+p_gw/mg_d)
    # * (1+p_aw/np.sqrt())
    th_r = p_gd_thre / (1+p_gw/mg_r)

    # down
    for x in range(size[0]-1):
        for y in range(size[1]):
            if difReg_d[x,y] :
                area = min(regMgr.getRegArea((x+1,y)),regMgr.getRegArea((x,y)))
                if area==0:
                    print("randomSeg.mergeRegion_A_2: area=0")
                if chara_d[x,y]<=th_d[x,y]*(1+(p_aw-1)*(a_bottom/area)**2):
                    regMgr.mergeRegion((x+1,y),(x,y))
    # right
    for x in range(size[0]):
        for y in range(size[1]-1):
            if difReg_r[x,y]:
                area = min(regMgr.getRegArea((x,y+1)),regMgr.getRegArea((x,y)))
                if area==0:
                    print("randomSeg.mergeRegion_A_2: area=0")
                if chara_r[x,y]<=th_r[x,y]*(1+(p_aw-1)*(a_bottom/area)**2):
                    regMgr.mergeRegion((x,y+1),(x,y))

    regMgr.settleRegion()


def mergeLittleRegion(dataMgr):
    threhold = dataMgr.para.p_la_bottom
    regMgr = dataMgr.regMgr
    chara0 = toMean(dataMgr)
    chara = np.sum(abs(chara0[:-1]-chara0[1:]),axis=2),np.sum(abs(chara0[:,:-1]-chara0[:,1:]),axis=2)
    size = dataMgr.shape

    for x0 in range(regMgr.shape[0]):
        for y0 in range(regMgr.shape[1]):
            while regMgr.getRegArea((x0,y0)) <= threhold:
            # if regMgr.regSumList[idx] <= threhold:
                pos = (x0,y0)
                reg1 = regMgr.getRegion(pos)
                new_reg_pos = regMgr.findNearReg(chara,reg1)
                                
                
                if new_reg_pos==0:
                    print("find reg fail!...")
                    return
                # if dif_min==0:
                #     print("? ...")
                #     return
                if not regMgr.mergeRegion(pos,new_reg_pos):
                    print("The same reg!..." )

    for x0 in range(regMgr.shape[0]):
        for y0 in range(regMgr.shape[1]):

            if regMgr.getRegArea((x0,y0)) <= threhold:
                print ("Little reg ...")
    
    return
         

def getLargeSegment(dataMgr, num = 1000, move_step = 4, killTinyReg = False):
    '''
    p_seg_color: cbcr gain
    p_cha_wc1: dist gain
    '''
    p_seg_color=dataMgr.para.p_seg_color
    p_seg_dist=dataMgr.para.p_seg_dist
    p_reg_threhold = dataMgr.para.p_la_bottom

    size = dataMgr.shape
    region = dataMgr.regMgr.space.copy()

    dist_max = p_seg_dist*(size[0]+size[1])+255+p_seg_color*(255*2)
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
  
    dataMgr.setGrad(p_seg_color)
    
    AdjData = dataMgr.Cdata
    cy_arr = AdjData[:,:,0]
    cb_arr = AdjData[:,:,1]
    cr_arr = AdjData[:,:,2]

    ## decided init center
    for i in range(num):
        ox,oy = random.randint(0,size[0]-1),random.randint(0,size[1]-1)
        ppos = (ox,oy)
        for j in range(move_step):
            minGrad = dataMgr.grad[ppos[0],ppos[1]]
            npos = ppos
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
    prepos = (-1,-1)

    for i in range(num):
        ## get dist
        (pos_x,pos_y,c_y,c_b,c_r) = sortedList[i]
        if (pos_x,pos_y) == prepos:
            continue
        prepos = (pos_x,pos_y)
        r_x1, r_x2, r_y1, r_y2 = getRange(pos_x,pos_y)
        x_dist = x_arr[r_x1:r_x2, r_y1:r_y2]-pos_x
        y_dist = y_arr[r_x1:r_x2, r_y1:r_y2]-pos_y
        cy_dist = cy_arr[r_x1:r_x2, r_y1:r_y2]-c_y
        cb_dist = cb_arr[r_x1:r_x2, r_y1:r_y2]-c_b
        cr_dist = cr_arr[r_x1:r_x2, r_y1:r_y2]-c_r

        dist_tem = np.sqrt( p_seg_dist*(np.square(x_dist) + np.square(y_dist)) + \
            np.square(cy_dist) + p_seg_color*(np.square(cb_dist) + np.square(cr_dist)) )
        
        for x in range(r_x1,r_x2):
            for y in range(r_y1, r_y2):
                if dist_tem[x-r_x1,y-r_y1] < dist[x,y]:
                    old_reg = region[x,y]
                    dist[x,y] = dist_tem[x-r_x1,y-r_y1]
                    region[x,y] = i+1

    region_visited = np.zeros(size[0:2], dtype = bool)
    
    for i in range( size[0]):
        for j in range(size[1]):
            if region_visited[i,j]:
                continue
            reg1, r_area1 = getRegDFS(region, (i,j),True,True)
            for pos1 in reg1:
                region_visited[pos1] = True
            
            if killTinyReg and r_area1<=p_reg_threhold:
                dataMgr.regMgr.labelLittleReg(reg1)
                # else:
                #     dataMgr.regMgr.addRegion(reg1) #np.zeros((size[0],size[1]),bool)
            else:
                dataMgr.regMgr.addRegion(reg1)
    chara = getChara2(dataMgr.Cdata,dataMgr.para)
    if killTinyReg:
        dataMgr.regMgr.mergeLabelReg(chara,p_reg_threhold)
        dataMgr.regMgr.settleRegion()

    # print('test1')
    # print(dataMgr.regMgr.vertifyFull())
    # print(dataMgr.regMgr.vertifyArea())
    
    if not dataMgr.regMgr.vertifyFull():
        print("???")
    if not dataMgr.regMgr.vertifyArea():
        print("???")
    # print('test2')
    # print(dataMgr.regMgr.vertifyFull())
    # print(dataMgr.regMgr.vertifyArea())
    return region



