import numpy as np
from util import nodelib, imglib
from seg import seglib
# from seg.region import *
from seg.randomSeg import *
import time

in_path_str = "Pic/"

DEBUG = False
TO_IMG = True
# DEF_LDATA = False #True


def setPara(dataMgr, gray, shape):
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
    pa = dataMgr.para


    if not DEF_LDATA:
        # pa.p_seg_color = 2
        # pa.p_seg_dist = 0.5
        # pa.p_cha_wc1 = 1
        # pa.p_cha_wc23 = pa.p_cha_wc1*pa.p_seg_color
        # pa.p_cha_we = 0.8
        # pa.p_cha_wr = 1
        # pa.p_cha_thre = 25
        # pa.p_la_bottom = 64
        # pa.p_la_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.shape[0]*dataMgr.shape[1]/1600))
        # pa.p_gd_we = 1.0
        # pa.p_gd_wr = 1.2
        # pa.p_gd_thre = 20
        # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top)
        # p_la_top = 400
        pass
    if DEF_LDATA:
        if not gray:
            pa.p_seg_color = 4
            pa.p_seg_dist = 0.5
            pa.p_cha_wc1 = 1
            pa.p_cha_wc23 = pa.p_cha_wc1*pa.p_seg_color
            pa.p_cha_we = 1.5*0.4
            pa.p_cha_wr = 0.8*0.4
            pa.p_cha_thre = 12*0.6
            pa.p_la_bottom =  int(max(16,dataMgr.getarea()/6400))
            pa.p_la_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.getarea()/1600))
            pa.p_gd_pow = 0.4
            pa.p_gd_we = 9
            pa.p_gd_wr = 12
            pa.p_gd_thre = 20
            pa.p_wth_amp = 2
            pa.p_wth_area  = 2
            pa.p_wth_grad  = 25
            pa.p_wth_thre  = 20*0.1
            # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top)
        else: #if gray
            # print("gray")
            pa.p_seg_color = 0
            pa.p_seg_dist = 0.14
            pa.p_cha_wc1 = 1
            pa.p_cha_wc23 = pa.p_cha_wc1*pa.p_seg_color
            pa.p_cha_we = 1.5*0.4
            pa.p_cha_wr = 0.8*0.4
            pa.p_cha_thre = 7.5*0.6
            pa.p_la_bottom = int(max(16,dataMgr.getarea()/6400))
            pa.p_la_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.getarea()/1600))
            pa.p_gd_pow = 0.5
            pa.p_gd_we = 8
            pa.p_gd_wr = 12
            pa.p_gd_thre = 30
            pa.p_wth_amp = 2
            pa.p_wth_area  = 5
            pa.p_wth_grad  = 75
            pa.p_wth_thre  = 30*0.1
            # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top)

def processFile(data, test = False):
    dataMgr = DataMgr(data)
    
    # if not imglib.isGray(data):
    #     return 0

    # (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top) = dataMgr.para
    param = dataMgr.para
    b_gray = imglib.isGray(data)
    setPara(dataMgr,imglib.isGray(data), dataMgr.shape )
    
    if not test:
        
        
        oc = []
        oce = []
        oc.append(dataMgr.data)
        oce.append(dataMgr.Cdata)
        start_time = time.time()
        getLargeSegment(dataMgr, 2000,killTinyReg=True) ## TODO
        output0 = dataMgr.region_copy()
        if np.mean(output0)!=np.mean(dataMgr.regMgr.space):
            print("Wrong!")
        print("RandomSeg edge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))    

        # ----previous merge                  --- #
        start_time = time.time()
        meanmergeRegionAdj(dataMgr,0.01)
        # mergeRegionAdj(dataMgr, 0.01)
        print("preMerge time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))        
        # # ------------------------------------- #

        # ---- merge Chara2                   --- #
        start_time = time.time()
        mergeRegion_A_2(dataMgr)
        print("Merge2 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))        
        # # ------------------------------------- #
        # # ---- merge Chara2                 --- #
        # start_time = time.time()
        # mergeRegion_A_2(dataMgr)
        # print("Merge2 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        # oc.append(toMean(dataMgr,False))
        # oce.append(toMean(dataMgr,True,b_gray))        
        # ------------------------------------- #
        # ---- merge Chara31                --- #
        start_time = time.time()
        mergeRegion_AG_31(dataMgr)
        print("Merge31 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))

        # ===================================== #
        print("Region num:\t\t\t--- %8d regions ---" %(dataMgr.regMgr.idxNum ))
        outRow1 = imglib.mergeArray(tuple(oc),axis=1, interval=20)
        outRow2 = imglib.mergeArray(tuple(oce),axis=1, interval=20)
        # out2 = imglib.mergeArray(tuple(list2),axis=1, interval=20)
        
        
        del dataMgr
        return imglib.mergeArray((outRow1,outRow2),axis=0, interval=20)
    else:
        pass