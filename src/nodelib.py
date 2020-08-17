
import numpy as np

def  getNearNode(x,y,size_x, size_y):
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

def getEdge(data):
    data_f = data
    
    l = np.empty_like(data_f)
    r = np.empty_like(data_f)
    u = np.empty_like(data_f)
    d = np.empty_like(data_f)
    l[:,:-1] = data_f[:,1:]
    l[:,-1] = data_f[:,-1]
    d[:-1] = data_f[1:]
    d[-1] = data_f[-1]
    r[:,1:] = data_f[:,:-1]
    r[:,0] = data_f[:,0]
    u[1:] = data_f[:-1]
    u[0] = data_f[0]
    l = np.not_equal(l,data_f)
    r = np.not_equal(r,data_f)
    u = np.not_equal(u,data_f)
    d = np.not_equal(d,data_f)
    out = np.logical_or(np.logical_or(l,r),np.logical_or(u,d))

    return out