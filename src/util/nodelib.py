
import numpy as np
from util import color, imglib

def arr_equal(a1,a2):
    if a1.shape != a2.shape:
        return None
    if len(a1.shape) != 1:
        return None
    for i in range(a1.shape[0]):
        if a1[i] != a2[i]:
            return False
    return True

def toColor(data, drawEdge = True):

    size_x, size_y = data.shape
    # if c!=3:
    #     print("?????")
    #     return None
    out = np.zeros((size_x, size_y, 3))
    # print(segment.shape)
    for x in range(size_x):
        for y in range(size_y):
            # print(seg_array)
            if data[x,y] !=0:
                out[x,y] = np.array(color.getHue(data[x,y]*10))
    if drawEdge:
        edge = getEdge(data)

        # out[edge] = [255,255,255]
        out[edge] = [0,0,0]

    return out
def toEdge(data, drawEdge = True):

    size_x, size_y = data.shape
    # if c!=3:
    #     print("?????")
    #     return None
    out = np.ones((size_x, size_y, 3))*255
    # print(segment.shape)
    # for x in range(size_x):
    #     for y in range(size_y):
    #         # print(seg_array)
    #         if data[x,y] !=0:
    #             out[x,y] = np.array(color.getHue(data[x,y]*10))
    if drawEdge:
        edge = getEdge(data)

        # out[edge] = [255,255,255]
        out[edge] = [0,0,0]

    return out

def combineEdge(datas, gray = True, toImg = True):

    # print(datas[0].shape)
    size_total = datas[0].shape
    # if len(size_total) == 3:
    #     size_x, size_y, _ = size_total
    # else:
    if len(size_total) ==3:
        print("In nodelib.combineEdge : data size is 3d.")
    size_x, size_y = size_total
    out = np.zeros((size_x, size_y))
    # for x in range(size_x):
    #     for y in range(size_y):
    #         # print(seg_array)
    #         if data[x,y] !=0:
    #             out[x,y] = np.array(color.getHue(data[x,y]*10))
    edges = []
    for data in datas:
        edges.append(getEdge(data))

        # out[edge] = [255,255,255]
        # out[edge] = [0,0,0]
    if not gray:
        for edge in edges:
            out = np.logical_or(out,edge)
        if not toImg:
            return out
        out *= 255
        return imglib.arrToImg(out)

    for edge in edges:
        out = out+edge
    if not toImg:
        return out/len(datas)
    out *= 255/len(datas)
    return imglib.arrToImg(out)
    

    return out

def getNearNode(x,y,size_x, size_y):
    if x<0 or x>=size_x:
        return 0
    if y<0 or y>=size_y:
        return 0

    l,r,u,d = True,True,True,True
    if x==0:
        u = False
    elif x == size_x-1:
        d = False
    if y==0:
        l = False
    elif y == size_y-1:
        r = False
          
    nodes = []

    if l:
        nodes.append((x,y-1))
    if r:
        nodes.append((x,y+1))
    if d:
        nodes.append((x+1,y))
    if u:
        nodes.append((x-1,y))
    return nodes



def getFrame(img,neg = False):
    if not neg:
        img_f = img
    else:
        img_f = np.logical_not(img)
    l = np.empty_like(img_f)
    r = np.empty_like(img_f)
    u = np.empty_like(img_f)
    d = np.empty_like(img_f)
    l[:,:-1] = img_f[:,1:]
    l[:,-1] = img_f[:,-1]
    d[:-1] = img_f[1:]
    d[-1] = img_f[-1]
    r[:,1:] = img_f[:,:-1]
    r[:,0] = img_f[:,0]
    u[1:] = img_f[:-1]
    u[0] = img_f[0]
    out = np.logical_or(np.logical_or(l,r),np.logical_or(u,d))

    return np.logical_and (np.logical_not(img_f), out)

def getEdge(data, neg = False):
    data_f = data
    
    # l = np.empty_like(data_f)
    r = np.empty_like(data_f)
    # u = np.empty_like(data_f)
    d = np.empty_like(data_f)
    # l[:,:-1] = data_f[:,1:]
    # l[:,-1] = data_f[:,-1]
    d[:-1] = data_f[1:]
    d[-1] = data_f[-1]
    r[:,1:] = data_f[:,:-1]
    r[:,0] = data_f[:,0]
    # u[1:] = data_f[:-1]
    # u[0] = data_f[0]
    # l = np.not_equal(l,data_f)
    r = np.not_equal(r,data_f)
    # u = np.not_equal(u,data_f)
    d = np.not_equal(d,data_f)
    # out = np.logical_or(np.logical_or(l,r),np.logical_or(u,d))
    out = np.logical_or(r,d)
    if neg:
        return np.logical_not(out)

    return out

def getGradient(data, unit_len = 1, b_return_vector_form = False):
    data_f = data
    r = np.empty_like(data_f)
    d = np.empty_like(data_f)
    d[:-1] = data_f[1:]
    d[-1] = data_f[-1]
    r[:,1:] = data_f[:,:-1]
    r[:,0] = data_f[:,0]
    gx = (d-data_f)/unit_len
    gy = (r-data_f)/unit_len
    if b_return_vector_form:
        return gx,gy #TODO

    grad = np.sqrt(np.square(gx) + np.square(gy))
    grad[-1] = grad[-2]
    grad[:,-1] = grad[:,-2]
    # print(grad.shape)
    return grad

def changeRegionNum(data,x,y,n, visited = None, n0 = -1):
    b_v = False
    if type(visited) != type(None):
        b_v = True
        
    size = data.shape
    if len(size) ==3:
        print("In nodelib.changeRegionNum : data size is 3d.")
    if n0==-1:
        n0 = data[x,y]
    if n==data[x,y]:
        return
    plist = []
    plist.append((x,y))
    while plist:
        x,y = plist.pop()
        if b_v:
            visited[x,y] += 1
        p_near = getNearNode(x,y,size[0],size[1])
        for point in p_near:
            nx,ny = point
            if data[nx,ny] == n0:
                data[nx,ny] = n
                plist.append(point)

    return







