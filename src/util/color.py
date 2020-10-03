"""
color.py
"""
import numpy as np
# def mix(c1,c2):
#     if c1.shape != (3,):
#         return 0
#     if c2.shape != (3,):
#         return 0
#     def harmonicMean(a,b):
#         return 2/(1/a+1/b)
#     c = np.array([0,0,0])
#     for i in range(3):
#         if c1[i] == 0 or c2[i] == 0:
#             c[i] = 0
#         else:
#             c[i] = harmonicMean(c1[i],c2[i])
#     return c
def getHue(color, isInt = True):
    if isInt:
        r,g,b = getHueRaw(color)
        return (int(r),int(g),int(b))
    else :
        return getHueRaw(color)

def getHueRaw(color):
    c = color%360
    if c<0 :
        print("???")
        return 0

    if c<60:
        return (255,255*c/60,0)
    elif c<120:
        return (255*(120-c)/60,255,0)
    elif c<180:
        return (0,255,255*(c-120)/60)
    elif c<240:
        return (0,255*(240-c)/60,255)
    elif c<300:
        return (255*(c-240)/60,0,255)
    # elif c<360:
    else:
        return (255,0,255*(360-c)/60)

# def color_gamma(x,b_255 = True):

    # g_data = np.empty_like( color_gamma)
    # r,g,b = data
    # if b_255:
    #     x0 = x/255.0
    # else:
    #     x0 = x
    # if x0>1 or x0<0 :
    #     print("Wrong!")
    #     return None
    # if x0>0.04045:
    #     return ((x+0.055)/1.055)**2.4
    # else:
    #     return x/12.92

