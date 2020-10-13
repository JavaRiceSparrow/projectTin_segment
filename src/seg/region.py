
import numpy as np

DEF_LDATA = False #True

class RegionMgr(object):
    def __init__(self, size0, size1):
        self.shape = (size0, size1)
        self.IntMatrix = np.zeros([self.shape[0],self.shape[1]],int)
        self.regSumList = [-1]
        self.regBoolList = [0]
        self.regNum = 0
        self.regSize = 0
    '''
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

    ''' 
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
        self.regBoolList.append(region)
        self.regSumList += np.sum(region)
        if (np.sum(self.IntMatrix[region]) != 0):
            print("\"region.py.RegionMgr.addRegion\": regions might be covered.")
        self.IntMatrix[region] = self.regSize

        # region
        return True
        

    def mergeRegion(self, pos1, pos2):
        x1,y1 = pos1
        x2,y2 = pos2
        reg1 = self.IntMatrix[x1,y1]
        reg2 = self.IntMatrix[x2,y2]
        # c1 = self.Ydata[x1,y1]
        # c2 = self.Ydata[x2,y2]
        # print("p1: (",c1[0],',',c1[1],',',c1[2],'), p2:(',c2[0],',',c2[1],',',c2[2],')')
        if reg1 == reg2:
            return
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
        return
        
class DataMgr(object):
    
    # dirList = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]

    def __init__(self, data):
        self.data = data
        self.Ydata = imglib.RGBtoYCbCr(data)
        self.Ldata = imglib.RGBtoLAB(data)
        self.shape = data.shape
        # self.size = (self.shape[0],self.shape[1])
        # the int array of region
        self.region = RegionMgr(self.shape[0],self.shape[1])
        # self.region.IntMatrix = np.zeros([self.size[0],self.size[1]],int)
        # self.regSumList = []
        # self.regBoolList = []
        # self.regNum = 0
        # self.regSize = 0


        
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

    def getNodeIdx(self,pos):
        x,y = pos
        return self.region.IntMatrix[x,y]
    # def getNodeIdx(self,x,y):
    #     return self.region.IntMatrix[x,y]

    

            