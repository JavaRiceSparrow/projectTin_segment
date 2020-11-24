
import numpy as np
from util import imglib, nodelib

DEF_LDATA = False 
DEF_LDATA = True

# ----------------------------------------------------- #
# Belong to getDFSRegions 
def getStartNode(region):
    size = region.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if region[i,j]:
                return (i,j)

def getRegDFS(region,node,b_list=False, b_area=False):
    x,y = node
    value = region[x,y]
    size = region.shape

    reg2 = np.zeros(region.shape,dtype=bool)
    reg2[x,y] = True
    if b_list:
        regList = [node]
    area = 0

    pos_stack=[node]
    # del node
    while(len(pos_stack) !=0):
        node1 = pos_stack.pop()
        nx,ny = node1
        nList = nodelib.getNearNode(nx,ny,size[0],size[1])
        area+=1
        for node2 in nList:
            if not reg2[node2] and region[node2]==value:
                pos_stack.append(node2)
                reg2[node2] = True
                if b_list:
                    regList.append(node2)

        del nList, node1, nx,ny
    if not b_list:
        if not b_area:
            return reg2
        else:
            return reg2, area
    else:
        if not b_area:
            return regList
        else:
            return regList, area

def getEdgeDFS(region,node):
    x,y = node
    value = region[x,y]
    size = region.shape

    reg2 = np.zeros(region.shape,dtype=bool)
    reg2[x,y] = True
    i_list = []
    o_list = []
    # area = 1

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
            if region[node2]!=value:
                i_list.append(node1)
                o_list.append(node2)

        del nList, node1, nx,ny
    return i_list,o_list

def rL2reg(rlist, shape):
    reg = np.zeros(shape, dtype=bool)
    for pos in rlist:
        reg[pos] = True
    return reg

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
        self.p_la_bottom    = 0
        self.p_la_top       = 0
        self.p_gd_we        = 0
        self.p_gd_wr        = 0
        self.p_gd_pow       = 0
        self.p_gd_thre      = 0
        self.p_wth_thre     = 0
        self.p_wth_area     = 0
        self.p_wth_grad     = 0

        # self.p_seg = {'w_color' : 4,'w_dist' : 0.5}
        # self.p_chara = {'w_c1' : 1, 'w_c23' : 1*4, 'w_edge' : 0.8, 'w_ridge' : 1, 'threhold' : 15}
        # self.p_mergeArea = {'bottom' : 0, 'top':0}
        # self.p_grad = {'weight': 0,"pow" :0, 'threhold' : 0}


class RegionMgr(object):
    def __init__(self, size0, size1):
        self.shape = (size0, size1)
        self.space = np.zeros([self.shape[0],self.shape[1]],int)
        self.areaList = [-1]
        # self.areaMat = np.zeros([self.shape[0],self.shape[1]],int)
        # self.areaList = []
        self.idxNum = 0
        self.idxMax = 0
        # self.mergecount = 0
    def getRegArea(self,pos):
        return self.areaList[self.space[pos]]
    def getRegIdx(self,pos):
        return self.space[pos]
    def copy(self):
        new_reg = RegionMgr(self.size0,self,size1) 
        new_reg.space = self.space.copy()
        new_reg.areaList = self.areaList.copy()
        new_reg.idxMax = self.idxMax
        new_reg.idxNum = self.idxNum
        return new_reg


    def addRegion(self, regList):
        # if region.dtype != bool:
        #     return False

        if len(regList)==0:
            return False
        self.idxMax += 1
        self.idxNum += 1
        r_sum = len(regList)
        self.areaList.append(r_sum)
        for pos in regList:
            if self.space[pos] != 0:
                print("\"region.RegionMgr.addRegion\": regions might be covered.")
            self.space[pos] = self.idxMax
            # self.areaMat[pos] = r_sum
        if len(self.areaList)-1 != self.idxMax:
            print("\"region.RegionMgr.addRegion\": Region is not in tail.(?)")

        # region
        return True

    # Not used
    # def cutRegion(self,idx , region):
    #     if region.dtype != bool:
    #         return False

    #     reg0 = self.space==idx
    #     if np.sum(np.logical_and(reg0,region)) != np.sum(region):
    #         print("\"region.RegionMgr.cutRegion\": regions out of original reg.")
    #     self.space[region] = 0
    #     self.areaMat[reg0] = self.areaMat[reg0][0]-np.sum(region)
    #     # self.regSumList[idx] -= np.sum(region)
    #     # self.regBoolList[idx][region] = False
    #     # if self.regSumList[idx] != np.sum(self.regBoolList[idx]):

    #     return self.addRegion(region)
        
    def delRegion(self, pos):
        reg0 = self.getRegion(pos,True)
        idx0 = self.space[pos]
        if idx0 == 0:
            print("region.delReg: ???")
            return False
        self.areaList[idx0] = 0
        for pos1 in reg0:
            self.space[pos1] = 0
            # self.areaMat[pos1] = 0
        self.idxNum -= 1
        # self.vertifyList()

    def mergeRegion(self, pos1, pos2):
        if type(pos1) == int or type(pos1) == np.int64:
            return False
            idx1 = pos1
            idx2 = pos2
        else:
            x1,y1 = pos1
            x2,y2 = pos2
            idx1 = self.space[x1,y1]
            idx2 = self.space[x2,y2]
        if idx1 == idx2:
            return False
        a1 = self.areaList[idx1]
        a2 = self.areaList[idx2]
        # TODO
        if a1 < a2:
            reg1 = self.getRegion(pos1,True)
            for pos1 in reg1:
                self.space[pos1] = idx2
            self.areaList[idx1] = 0
            self.areaList[idx2] = a1+a2   
            # self.mergecount += a1
        else:
            reg2 = self.getRegion(pos2,True)
            for pos2 in reg2:
                self.space[pos2] = idx1
            self.areaList[idx1] = a1+a2
            self.areaList[idx2] = 0
            # self.mergecount += a2
                 
            
        self.idxNum -= 1
        return True

    def settleRegion(self):
        return 
        pass
        if not self.vertifyFull():
            print("region.settleRegion: not vertifyFull")
        if not self.vertifyArea():
            print("region.settleRegion: not vertifyArea")
        # numList = np.zeros([self.idxMax], dtype=bool)
        idxList = np.zeros([self.idxMax], dtype=bool)
        # for i in range(self.shape[0]):
        #     for j in range(self.shape[1]):
        #         idx = self.space[i,j]
        #         numList[idx-1] = 1
        # idx = 0
        nidx = 0
        for idx in range(self.idxMax):
            if self.areaList[idx+1]>0:
                nidx += 1
                idxList[idx] = nidx
                self.areaList[nidx] = self.areaList[idx+1]
        self.areaList = self.areaList[:nidx+1]
        self.idxMax = self.idxNum = nidx
            # idx += 1

        # for i in range(len(numList)):
        #     if numList[i]:
        #         idx = idx+1
        #         idxList[i] = idx
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                idx = self.space[i,j]
                if idxList[idx-1]<=0:
                    print("region.settleRegion: label or empty region exist...")
                    return
                self.space[i,j] = idxList[idx-1]
        

        
        self.vertifyFull()
        self.vertifyArea()

    def getRegion(self, pos, b_list=True):
        # x,y = pos
        
        return getRegDFS(self.space, pos, b_list=b_list)

    def vertifyFull(self):
        if np.min(self.space)==0:
            return False
        if np.sum(self.areaList[1:])!=self.shape[0]*self.shape[1]:
            print("Area not full: ", np.sum(self.areaList[1:]),"/",self.shape[0]*self.shape[1])
            return False

        return True

    def vertifyArea(self):
        visited = np.zeros(self.shape,dtype=bool)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if not visited[i,j]:
                    rlist, area = getRegDFS(self.space, (i,j), b_list=True, b_area=True)
                    idx = self.space[i,j]
                    if area!=self.areaList[idx]:
                        print("Area not Vertified ... ")
                        # print(self.space)
                        # print(area, ", ",self.areaList[idx])
                        return False
                    for pos in rlist:
                        visited[pos] = True

        return True

    def vertifyCount(self):
        count = 0
        region_visited = np.zeros(self.shape,dtype=bool)
        for i in range( size_x):
            for j in range(size_y):
                if region_visited[i,j]:
                    continue
                # idx = regMgr.space[i,j]
                reg = regMgr.getRegion((i,j))
                for pos in reg:
                    region_visited[pos] = True
                # if np.sum(reg)==0:
                #     continue
                count += 1
        if count != self.idxNum:
            return False
        # TODO

        return True


    def labelLittleReg(self, regList):
        # if self.space[regList[0]]!=0:
        #     self.delRegion(regList[0])
        for pos in regList:
            if self.space[pos]!=0:
                print("region.labelLittleReg: wrong region")
                return False
            self.space[pos] = -2
        return True
        # self.regSumList[idx]=0
    def findNearReg(self, chara, regList, func = False):
        reg1 = rL2reg(regList, self.shape)
        nlist_i, nlist_o = getEdgeDFS(reg1,regList[0])
        dif_min = 100000
        new_reg_pos = 0
        for i in range(len(nlist_i)):
            pos_i = nlist_i[i]
            pos_o = nlist_o[i]
            # nlist= nodelib.getNearNode(x,y,size[0],size[1])
            if pos_i[1]==pos_o[1]:
                pos_f = (min(pos_i[0],pos_o[0]),pos_i[1])
                if chara[0][pos_f]<dif_min:
                    new_reg_pos = pos_o
                    dif_min = chara[0][pos_f]
            elif pos_i[0]==pos_o[0]:
                pos_f = (pos_i[0],min(pos_i[1],pos_o[1]))
                if chara[1][pos_f]<dif_min:
                    new_reg_pos = pos_o
                    dif_min = chara[1][pos_f]

        
            if new_reg_pos == 0:
                print("findNearReg fail...")
                return
        return new_reg_pos
    
    def mergeLabelReg(self, chara, threhold, test=False):

        reg = (self.space==0)
        if np.sum(reg) !=0:
            print("region.mergeLabelReg: fatal err...")
            return False
        # print("Start")
        
        reg = (self.space==-2)
        # regsum = np.sum(reg)
        if np.sum(reg)==0:
            return True
        del reg

        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if not self.space[i,j]==-2:
                    continue
                
                rList1 = getRegDFS(self.space,(i,j),True)
                # old_reg_pos = rList1[0]
                # self.delRegion(old_reg_pos)
                for pos in rList1:
                    self.space[pos] = 0
                self.addRegion(rList1)

                if test and len(rList1) <= threhold:
                    new_reg_pos = self.findNearReg(chara, rList1)
                    if type(new_reg_pos)==int:
                        print("region.mergeLabelReg: ???")
                    
                    if not self.mergeRegion(old_reg_pos,new_reg_pos):
                        print("region.mergeLabelReg: The same reg!..." )
                # for pos1 in rList1:
                #     reg[pos1] == 0
                # regsum -= np.sum(reg1)

        self.settleRegion()
        return True




            
        
class DataMgr(object):
    
    # dirList = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]

    def __init__(self, data):
        self.data = data
        self.Ydata = imglib.RGBtoYCbCr(data)
        self.Ldata = imglib.RGB2Lab(data)
        self.shape = data.shape[0:2]
        
        self.grad = 0.0
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
    def getarea(self):
        return int(self.shape[0]*self.shape[1])

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
        if not self.regMgr.vertifyFull():
            print('Region is not full!!!')
            return 0

        self.meanGrad = np.ones(self.shape,dtype = float)*-1.0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):  
                if self.meanGrad[i,j]==-1:
                    
                    reg = self.regMgr.getRegion((i,j),True)
                    grad_sum = 0
                    for pos in reg:
                        grad_sum += self.grad[pos]
                    grad_mean = grad_sum/len(reg)
                    for pos in reg:
                        self.meanGrad[pos] = grad_mean
        if np.min(self.meanGrad)<0:
            print('Region is not full!!!')
            return 0
        # return self.regMgr.IntMatrix[x,y]
        return self.meanGrad



    
    

            