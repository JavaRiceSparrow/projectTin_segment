
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
        self.IntMatrix = np.zeros([self.shape[0],self.shape[1]],int)
        self.regSumList = [-1]
        self.regBoolList = [0]
        self.regNum = 0
        self.regSize = 0
    # '''
    def settleRegion(self):
        # TODO
        return 0

        endP_i = 1
        emptyP_i = endP_i
        test_a = np.zeros(self.size,dtype=bool)

        while endP_i <=self.regSize:
            if self.regSumList[endP_i] == 0:
                pass
            else:
                self.regSumList[emptyP_i] = self.regSumList[endP_i] 
                self.regBoolList[emptyP_i] = self.regBoolList[endP_i]
                test_a += self.regBoolList[endP_i]
                emptyP_i += 1
            endP_i += 1

        print(np.min(np.ones(self.size,dtype=bool)==test_a))

        self.regSumList = self.regSumList[:emptyP_i]
        self.regBoolList = self.regBoolList[:emptyP_i]
        self.regSize = emptyP_i-1
        self.regNum = emptyP_i-1
        # print(self.regSize,', ',len(self.regBoolList))

        test_b = np.zeros(self.size,dtype=bool)
        for i in range(1,self.regSize):
            test_b += self.regBoolList[i]
            self.region.IntMatrix[self.regBoolList[i]] = i
        print(np.min(np.ones(self.size,dtype=bool)==test_b))
        print(np.min(test_a==test_b))

    # ''' 
    def simpCheck(self):
        if self.regNum>self.regSize:
            return False
        if len(self.regSumList) != len(self.regBoolList):
            return False
        if self.regSumList[0] != -1:
            return False
        if self.regBoolList != 0:
            return False

    def addRegion(self, region):
        if region.dtype != bool:
            return False
        self.regNum += 1
        self.regSize += 1
        # print("Add reg ", self.regNum)
        self.regBoolList.append(region)
        self.regSumList.append(np.sum(region))
        if (np.sum(self.IntMatrix[region]) != 0):
            print("\"region.RegionMgr.addRegion\": regions might be covered.")
        self.IntMatrix[region] = self.regSize

        # region
        return True

    def cutRegion(self,idx , region):
        if region.dtype != bool:
            return False
        if np.sum(np.logical_and(self.regBoolList[idx],region)) != np.sum(region):
            print("\"region.RegionMgr.cutRegion\": regions out of original reg.")
        # sum1 = np.sum(region)
        self.IntMatrix[region] = 0
        # print(np.sum(self.IntMatrix[region]))
        self.regSumList[idx] -= np.sum(region)
        self.regBoolList[idx][region] = False
        # if self.regSumList[idx] != np.sum(self.regBoolList[idx]):
        #     print("?")


        # region
        return self.addRegion(region)
        

    def mergeRegion(self, pos1, pos2):
        if type(pos1) == int:
            reg1 = pos1
            reg2 = pos2
        else:
            x1,y1 = pos1
            x2,y2 = pos2
            reg1 = self.IntMatrix[x1,y1]
            reg2 = self.IntMatrix[x2,y2]
        # c1 = self.Ydata[x1,y1]
        # c2 = self.Ydata[x2,y2]
        # print("p1: (",c1[0],',',c1[1],',',c1[2],'), p2:(',c2[0],',',c2[1],',',c2[2],')')
        if reg1 == reg2:
            return False
        # TODO
        if self.regSumList[reg1]< self.regSumList[reg2]:
            self.IntMatrix[self.regBoolList[reg1]] = reg2
            # self.regBoolList[reg2] = np.logical_or(self.regBoolList[reg1], self.regBoolList[reg2])
            self.regBoolList[reg2][self.regBoolList[reg1]] = True
            self.regBoolList[reg1] = 0
            self.regSumList[reg2] += self.regSumList[reg1]
            self.regSumList[reg1] = 0
            # nodelib.mergeRegionMgr(self.region, x1,y1,reg2)
        else:
            self.IntMatrix[self.regBoolList[reg2]] = reg1
            self.regBoolList[reg1][self.regBoolList[reg2]] = True
            self.regBoolList[reg2] = 0
            self.regSumList[reg1] += self.regSumList[reg2]
            self.regSumList[reg2] = 0
            # nodelib.mergeRegionMgr(self.region, x2,y2,reg1)
        self.regNum -= 1
        return True

    def vertifyList(self):
        count = 0
        su = True
        for i in range(1,self.regSize+1):
            if self.regSumList[i]==0:
                continue
            count += 1
            if self.regBoolList[i].dtype != bool:
                su = False
            if np.sum(self.regBoolList[i]) != self.regSumList[i]:
                self.regSumList[i] = np.sum(self.regBoolList[i])
                su = False
        if count != self.regNum:
            su = False

        return su
    def vertifyRegion(self):
        count = 0
        sum1 = self.shape[0]*self.shape[1]
        su = True
        reg = np.zeros(self.shape,dtype=bool)
        for i in range(1,self.regSize+1):
            if self.regSumList[i]==0:
                continue
            count += 1
            if np.sum(np.logical_and(reg,self.regBoolList[i]))!=0:
                su = False
                return False
                break
            reg[self.regBoolList[i]] = True
            
        if count != self.regNum:
            su = False
        if np.sum(reg) != sum1:
            return False

        return su

    def mergeLittleReg(self, idx,threhold):
        reg1 = self.regBoolList[idx]
        def getStartNode(region):
            size = region.shape
            for i in range(size[0]):
                for j in range(size[1]):
                    if region[i,j]:
                        return (i,j)
        def getEndNode(region):
            size = region.shape
            for j in range(size[1]-1,-1,-1):
                for i in range(size[0]-1,-1,-1):
                    if region[i,j]:
                        return (i,j)
        x,y = getStartNode(reg1)
        if x!=0 :
            self.mergeRegion((x,y),(x-1,y))
        elif y!=0:
            self.mergeRegion((x,y),(x,y-1))
        else:
            x,y = getEndNode(reg1)
            self.mergeRegion((x,y),(x,y+1))

            
        
class DataMgr(object):
    
    # dirList = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]

    def __init__(self, data):
        self.data = data
        self.Ydata = imglib.RGBtoYCbCr(data)
        self.Ldata = imglib.RGB2Lab(data)
        self.shape = data.shape[0:2]
        
        self.grad = 0
        self.para = dataParam

        self.region = RegionMgr(self.shape[0],self.shape[1])

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
    # def set_region(self, region):
    #     self.region.IntMatrix = region
    def region_copy(self):
        return self.region.IntMatrix.copy()
    

    def getNodeIdx(self,pos):
        x,y = pos
        return self.region.IntMatrix[x,y]
    # def getNodeIdx(self,x,y):
    #     return self.region.IntMatrix[x,y]

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
        if not self.region.vertifyRegion():
            print("?_?_?_?")
        meanGrad = np.zeros(self.shape,dtype = float)
        for i in range(1,self.region.regSize+1):
            
            if self.region.regSumList[i] == 0:
                continue
            reg = self.region.regBoolList[i]
            meanGrad[reg] = np.mean(self.grad[reg])
        # return self.region.IntMatrix[x,y]
        return meanGrad

    
    

            