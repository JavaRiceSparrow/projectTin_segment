import numpy as np
from util import nodelib, imglib

DEBUG = False
# DEBUG = True

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


def getRidge(data, sigma=0.4, dx=0.5, length=6):
    if len(data.shape) == 2:
        print("Wrong!")
        return None
        
    glist = []
    x_axis = (np.arange((length*2+1))-length)*dx
    x_2 = np.square(x_axis)
    hx = (4*sigma**2 * x_2-2*sigma) * np.exp(-sigma*x_2)
    hx -= np.mean(hx)
    # if np.mean(hx)!=0:
    #     print("W :" , np.mean(hx))

    for i in range(3):
        glist.append(matrixConvolution(data[:,:,i], hx, axis_along=0))
        glist.append(matrixConvolution(data[:,:,i], hx, axis_along=1))
    # print(glist)
    gsquare = np.square(glist.pop())
    while glist:
        gsquare = gsquare + np.square(glist.pop())

    return np.sqrt(gsquare)


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
        print("Filter size wrong! :", filter.shape)
        return None
    fsize = int(filter.shape[0])
    cdata = np.zeros((fsize,data.shape[0],data.shape[1]))
    mid = int((fsize-1)/2)
    if axis_along == 0:
        for i in range(mid):
            cdata[i,mid-i:] = data[:i-mid]
            cdata[i,:mid-i] = data[0:1]

        cdata[mid] = data
        for i in range(mid):
            # i = i-mid
            cdata[-i-1][:i-mid] = data[mid-i:] 
            cdata[-i-1][i-mid:] = data[-1:] 
    else :
        for i in range(mid):
            cdata[i,:,mid-i:] = data[:,:i-mid] 
            cdata[i,:,:mid-i] = data[:,0:1]

        cdata[mid] = data
        for i in range(mid):
            # i = i-mid
            cdata[-i-1,:,:i-mid] = data[:,mid-i:] 
            cdata[-i-1,:,i-mid:] = data[:,-1:] 
    
    for i in range(fsize):
        cdata[i] = cdata[i] * filter[i]
    
    return np.sum(cdata, axis = 0)



