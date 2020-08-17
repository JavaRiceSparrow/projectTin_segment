# Imglib.py


import numpy as np
from PIL import Image

# ImgSizeX = 189
# ImgSizeY = 189


#      -X
#       ^
#       |
# -Y<---O---> Y
#       |
#       V
#       X

def arrToImg(data):
    if data.dtype == 'float64' and np.max(data) == 1:
        ndata = np.around(255*data)
    else:
        ndata = data.copy()
    ndata = np.expand_dims(ndata, axis = 2)
    ndata = np.concatenate([ndata, ndata, ndata], axis=2).astype('uint8')
    return ndata

def img3dTo2d(data):
    if len(data.shape) != 3:
        return 0
    return (data[:,:,0] + data[:,:,1]+data[:,:,0])/3
    # .astype('uint8')

def mergeArray(datas, axis=1, interval = 0):
    '''
    '''
    data1 = datas[0]
    # print(data1.shape)
    # if len(data1.shape) !=3:
    #     return 0

    tdatas = tuple(datas)
    if interval != 0:
        list1 = []
        if axis == 0:
            itv = np.zeros((interval,data1.shape[1],3))
        elif axis == 1:
            itv = np.zeros((data1.shape[0],interval,3))
        for data in tdatas:
            list1.append(data)
            list1.append(itv)
        list1.pop()
        tdatas = tuple(list1)

    return np.concatenate((tdatas), axis = axis)

def getBiImg(path):
    data = getImg(path, Blight = True)

    return (data<=220) # *255
    # return data


def getImg(path, Blight = False, Black = False, to_3d = False):
    try:  
        img  = Image.open(path)  
    except IOError: 
        print(path + " is not a valid path.")
        return 0

    rawData = np.array(img)
    if to_3d:
        if len(rawData.shape) == 3:
            if rawData.shape[2] == 4:
                return rawData[:,:,:-1]
            return rawData
        elif len(rawData.shape) == 2:
            return arrToImg(rawData)
        else:
            return 0
    if Black:
        if len(rawData.shape) == 3:
            return np.around((rawData[:,:,0] + rawData[:,:,1]+rawData[:,:,0])/3)
        else :
            return rawData
    if not Blight:
        return rawData
    # print(rawData[10:30,40:60])
    if len(rawData.shape) == 3:
        brightData = rawData[:,:,0]*0.299+ \
            rawData[:,:,1]*0.587+rawData[:,:,0]*0.114
    else :
        brightData = rawData
    if Blight:
        return brightData

    # data = (brightData<=220) # *255
    # return data

    # edge = [data[1,:], data[1:-1:]]

def saveImg(data, path):
    if np.max(data) == 1:
        data = 255*data
        # data = 255*(1-data)
    if len(data.shape) == 2:
        data1 = np.expand_dims(data, axis = 2)

        bmpImg =  np.concatenate([data1, data1, data1], axis=2).astype('uint8')
    else:
        bmpImg = data.astype('uint8')

    # print(bmpImg.shape)
    # mp.image.imsave('result.bmp', bmpImg)
    im = Image.fromarray(bmpImg)
    im.save(path)

def showImg(data):
    if np.max(data) == 1:
        data = 255*data
        # data = 255*(1-data)
    if len(data.shape) == 2:
        data1 = np.expand_dims(data, axis = 2)
        bmpImg = np.concatenate([data1, data1, data1], axis=2).astype('uint8')
    else:
        bmpImg = data.astype('uint8')

    im = Image.fromarray(bmpImg)
    im.show()

def loadAllImg(charRange, wordRange, returnType = 2, Blight = False):
    #Path of the fifth word of the 4th dir is '1/4/base_1_5_4.bmp
    #Path of the fifth word of the 4th dir is '1/4/base_5_1_4.bmp
    str1 = 'data/1/'
    str2 = '/database/base_1_'
    str2n = '/testcase/word_'
    str3 = '_'
    str3n = '_1_'
    str4 = '.bmp'

    charNum = 15
    wordNum = 50

    bData, bTest = True, True

    if returnType == 1:
        bData = False
    if returnType == 0:
        bTest = False

    if bData:
        data = np.empty([charRange[1], wordRange[1], ImgSizeX, ImgSizeY], dtype = int)
    if bTest:
        test = np.empty([charRange[1], wordRange[1], ImgSizeX, ImgSizeY], dtype = int)

    # path_data = str1+charRange+str2+wordRange+str3

    for i in range(charRange[1]):
        for j in range(wordRange[1]):
            ni = i + charRange[0]
            nj = j + wordRange[0]
            if bData:
                path_data = str1+str(ni)+str2+str(nj)+str3+ str(ni) + str4
                d = getImg(path_data, Blight)
                if type(d) != int:
                   data[i,j] = d
                else :
                    return 0
            if bTest:
                path_test = str1+str(ni)+str2n+str(nj)+str3n+ str(ni) + str4
                t = getImg(path_test, Blight)
                if type(t) != int:
                    test[i,j] = t
                else :
                    return 0

    if returnType == 0:
        return data
    elif returnType == 1:
        return test
    else :
        return data, test

def loadTemplate(charIdx, temIdx = 1, Blight = False):
    str1 = 'data/template/tem' + str(temIdx) + '/'
    str3 = '.bmp'
    # data = np.empty([charRange[1], ImgSizeX, ImgSizeY], dtype = int)

    
    
    path_data = str1+str(charIdx)+str3
    data = getImg(path_data, Blight)
    if type(data) != int:
        return data
    else :
        return 0
        

    


def pointNorm(point, Img):
    x, y = Img.shape
    axis_x = np.sum(Img, axis = 1)
    axis_y = np.sum(Img, axis = 0)
    for i in range(x):
        if axis_x[i]:
            x_min = i
            break
    for i in range(y):
        if axis_y[i]:
            y_min = i
            break
    for i in range(x-1,-1,-1):
        if axis_x[i]:
            x_max = i
            break
    for i in range(y-1,-1,-1):
        if axis_y[i]:
            y_max = i
            break
    if x_max == x_min:
        return 0
    if y_max == y_min:
        return 0

    x0 = np.mean(x_max,x_min)
    y0 = np.mean(y_max,y_min)
    size = np.max(x_max-x_min, y_max-y_min)
    point_x, point_y = point
    newpoint = (100*(point_x-x0)/size,100*(point_y-y0)/size)

    if type(point) == tuple:
        return newpoint
    elif type(point) == np.ndarray:
        return np.array(newpoint)




# x = np.array([[1,0,1], [0,1,1],[0,1,0],[1,0,0]])
# print(moment(x,1,2))
