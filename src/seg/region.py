
import numpy as np
from util import imglib, nodelib

DEF_LDATA = False 
DEF_LDATA = True




class dataParam(object):
    '''
    (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top)
    p_seg_color : gain of cb&cr
    p_seg_dist : gain of dist
    p_cha_wc1 : gain of cy (cha)
    p_cha_wc23 : gain of cb&cr (cha)( = p_cha_wc1*p_seg_color )
    p_cha_we : weight of edge
    p_cha_wr : weight of ridge
    p_cha_thre : 25
    p_la_bottom : little area merge down
    p_la_top : little area merge up
    '''
    def __init__(self):
        self.p_seg_color    = 4
        self.p_seg_dist     = 0.5
        self.p_cha_wc1      = 1
        self.p_cha_wc23     = self.p_cha_wc1*self.p_seg_color 
        self.p_cha_we       = 0.8
        self.p_cha_wr       = 1
        self.p_cha_thre     = 15
        self.p_la_bottom = 0
        self.p_la_top    = 0
        self.p_gd_we        = 0
        self.p_gd_wr        = 0
        self.p_gd_pow       = 0
        self.pa.p_gd_thre   = 0

        # self.p_seg = {'w_color' : 4,'w_dist' : 0.5}
        # self.p_chara = {'w_c1' : 1, 'w_c23' : 1*4, 'w_edge' : 0.8, 'w_ridge' : 1, 'threhold' : 15}
        # self.p_mergeArea = {'bottom' : 0, 'top':0}
        # self.p_grad = {'weight': 0,"pow" :0, 'threhold' : 0}


class RegionMgr(object):
    def __init__(self, size0, size1):
        self.shape = (size0, size1)
        self.space = np.zeros([self.shape[0],self.shape[1]],int)
        self.areaMat = np.zeros([self.shape[0],self.shape[1]],int)
        # self.areaList = []
        self.idxNum = 0
        self.idxMax = 0


    def addRegion(self, region):
        if region.dtype != bool:
            return False

        if np.sum(region)==0:
            return False
        self.idxMax += 1
        if (np.sum(self.space[region]) != 0):
            print("\"region.RegionMgr.addRegion\": regions might be covered.")
        self.space[region] = self.idxMax
        self.areaMat[region] = np.sum(region)

        # region
        return True

    def cutRegion(self,idx , region):
        if region.dtype != bool:
            return False

        reg0 = self.space==idx
        if np.sum(np.logical_and(reg0,region)) != np.sum(region):
            print("\"region.RegionMgr.cutRegion\": regions out of original reg.")
        self.space[region] = 0
        self.areaMat[reg0] = self.areaMat[reg0][0]-np.sum(region)
        # self.regSumList[idx] -= np.sum(region)
        # self.regBoolList[idx][region] = False
        # if self.regSumList[idx] != np.sum(self.regBoolList[idx]):

        return self.addRegion(region)
        
    def delRegion(self, idx):
        reg0 = self.space==idx
        self.space[reg0] = 0
        self.areaMat[reg0] = 0
        self.idxNum -= 1    
        # self.vertifyList()

    def mergeRegion(self, pos1, pos2):
        if type(pos1) == int or type(pos1) == np.int64:
            idx1 = pos1
            idx2 = pos2
        else:
            x1,y1 = pos1
            x2,y2 = pos2
            idx1 = self.space[x1,y1]
            idx2 = self.space[x2,y2]
        if idx1 == idx2:
            return False
        reg1 = self.space==idx1
        reg2 = self.space==idx2
        a1 = self.areaMat[reg1][0]
        a2 = self.areaMat[reg2][0]
        # TODO
        if a1 < a2:
            self.space[reg1] = idx2
            self.areaMat[reg1] = a1+a2
            self.areaMat[reg2] = a1+a2   
        else:
            self.space[reg2] = idx1
            self.areaMat[reg1] = a1+a2
            self.areaMat[reg2] = a1+a2
            
        self.idxNum -= 1
        return True

    def settleRegion(self):
        if not self.vertifyFull():
            print("??? ??")
        numList = np.zeros([self.idxMax], dtype=bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                idx = self.space[i,j]
                numList[idx-1] = 1
        for i in range(len(numList)):
            if numList[i]:
                idx = i+1
                # TODO

    def getRegion(self, pos):
        # x,y = pos
        def getRegDFS(region,node):
            x,y = node
            reg2 = np.zeros(self.shape,dtype=bool)
            value = region[x,y]
            reg2[x,y] = True
            p_seg_color=[node]
            # del node
            size = region.shape
            while(len(p_seg_color) !=0):
                node1 = p_seg_color.pop()
                nx,ny = node1
                nList = nodelib.getNearNode(nx,ny,self.shape[0],self.shape[1])
                for node2 in nList:
                    if not reg2[node2] and region[node2]==value:
                        p_seg_color.append(node2)
                        reg2[node2] = True

                del nList, node1, nx,ny

            return reg2
        return getRegDFS(self.space, pos)

    def vertifyFull(self):
        if np.min(self.space)==0:
            return False
        if np.min(self.areaMat)==0:
            return False

        return su
    # def vertifyRegion(self):
    #     for i in range:

    # def mergeLittleReg(self, idx,threhold):
    #     if idx>self.idxMax:
    #         return False
    #         # TODO
    #     reg1 = self.space==idx
    #     if  threhold < np.sum(reg1):
    #         return 
    #     def getStartNode(region):
    #         size = region.shape
    #         for i in range(size[0]):
    #             for j in range(size[1]):
    #                 if region[i,j]:
    #                     return (i,j)
    #     def getEndNode(region):
    #         size = region.shape
    #         for j in range(size[1]-1,-1,-1):
    #             for i in range(size[0]-1,-1,-1):
    #                 if region[i,j]:
    #                     return (i,j)
    #     x,y = getStartNode(reg1)
    #     if x!=0 :
    #         self.mergeRegion((x,y),(x-1,y))
    #     elif y!=0:
    #         self.mergeRegion((x,y),(x,y-1))
    #     else:
    #         x,y = getEndNode(reg1)
    #         self.mergeRegion((x,y),(x,y+1))


    def labelLittleReg(self, region):
        idx = self.space[region][0]
        # reg1 = self.space==idx
        if np.sum(region) != self.areaMat[region][0]:
            print("lLR: Area wrong.")
            return False

        self.delRegion(idx)
        self.space[region] = -2
        # self.regSumList[idx]=0
    
    def mergeLabelReg(self, chara, threhold):
        def getStartNode(region):
            size = region.shape
            for i in range(size[0]):
                for j in range(size[1]):
                    if region[i,j]:
                        return (i,j)

        def getRegDFS(region,node):
            x,y = node
            reg2 = np.zeros(self.shape,dtype=bool)
            value = region[x,y]
            reg2[x,y] = True
            p_seg_color=[node]
            # del node
            size = region.shape
            while(len(p_seg_color) !=0):
                node1 = p_seg_color.pop()
                nx,ny = node1
                nList = nodelib.getNearNode(nx,ny,self.shape[0],self.shape[1])
                for node2 in nList:
                    if not reg2[node2] and region[node2]==value:
                        p_seg_color.append(node2)
                        reg2[node2] = True

                del nList, node1, nx,ny

            return reg2
        reg = self.space==-2
        regsum = np.sum(reg)
        if np.sum(reg)==0:
            return True
        

        while regsum!=0:
            pos = getStartNode(reg)
            reg1 = getRegDFS(reg,pos)
            self.space[reg1]=0
            self.addRegion(reg1)

            if np.sum(reg1) <= threhold:
            
                edge = nodelib.getInnerFrame(reg1)
                dif_min = 100000
                new_reg_idx = 0
                for x in range(self.shape[0]):
                    for y in range(self.shape[1]):
                        if edge[x,y]:
                            # nlist= nodelib.getNearNode(x,y,size[0],size[1])
                            if x!=0 and not reg1[x-1,y] and chara[0][x-1,y]<dif_min:
                                new_reg_idx = self.space[x-1,y]
                                dif_min = chara[0][x-1,y]
                            if y!=0 and not reg1[x,y-1] and chara[1][x,y-1]<dif_min:
                                new_reg_idx = self.space[x,y-1]
                                dif_min = chara[1][x,y-1]
                            if x!=size[0]-1 and not reg1[x+1,y] and chara[0][x,y]<dif_min:
                                new_reg_idx = self.space[x+1,y]
                                dif_min = chara[0][x,y]
                            if y!=size[1]-1 and not reg1[x,y+1] and chara[1][x,y]<dif_min:
                                new_reg_idx = self.space[x,y+1]
                                dif_min = chara[1][x,y]
                
                if new_reg_idx==0:
                    print("? ? ?")
                    return
                if dif_min==0:
                    print("? ...")
                    return
                if not self.mergeRegion(self.regSize,new_reg_idx):
                    print("The same reg!..." )
            reg[reg1] == 0
            regsum -= np.sum(reg1)
        return True




            
        
class DataMgr(object):
    
    # dirList = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]

    def __init__(self, data):
        self.data = data
        self.Ydata = imglib.RGBtoYCbCr(data)
        self.Ldata = imglib.RGB2Lab(data)
        self.shape = data.shape[0:2]
        
        self.grad = 0
        self.para = dataParam

        self.regMgr = RegionMgr(self.shape[0],self.shape[1])

        if DEF_LDATA:
            self.Cdata = self.Ldata
        else:
            self.Cdata = self.Ydata


    
    # def getsize(self):
    #     return self.size    
    def getshape(self):
        return self.shape

    def data(self):
        return self.data
    def data_copy(self):
        return self.data.copy()
    # def set_region(self, regMgr):
    #     self.regMgr.IntMatrix = regMgr
    def region_copy(self):
        return self.regMgr.space.copy()
    

    def getNodeIdx(self,pos):
        x,y = pos
        return self.regMgr.space[x,y]
    # def getNodeIdx(self,x,y):
    #     return self.regMgr.IntMatrix[x,y]

    def setGrad(self, p_seg_color=-1):
        if p_seg_color==-1:
            p_seg_color=self.para[0]

        # AdjData = dataMgr.Cdata
        cy_arr = nodelib.getGradient(self.Cdata[:,:,0])
        cb_arr = nodelib.getGradient(self.Cdata[:,:,1])
        cr_arr = nodelib.getGradient(self.Cdata[:,:,2])
        # grad_func = cy_arr + np.sqrt(p_seg_color)*(cb_arr+cr_arr)
        # nodelib.getGradient(cy_arr)
        self.grad = np.abs(cy_arr) + np.sqrt(p_seg_color)*(np.abs(cb_arr)+np.abs(cr_arr))
    def getMeanGrad(self):
        # if not self.regMgr.vertifyRegion():
        #     # print("?_?_?_?")
        #     pass
        meanGrad = np.ones(self.shape,dtype = float)*-1
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):  
                if meanGrad[i,j]!=-1:
                    
                    reg = self.regMgr.space==self.regMgr.space[i,j]
                    meanGrad[reg] = np.mean(self.grad[reg])
        # return self.regMgr.IntMatrix[x,y]
        return meanGrad

    
    

            