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
    (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_laMge_bottom,p_laMge_top)
    p_seg_color : gain of cb&cr
    p_seg_dist : gain of dist
    p_cha_wc1 : gain of cy (cha)
    p_cha_wc23 : gain of cb&cr (cha)( = p_cha_wc1*p_seg_color )
    p_cha_we : weight of edge
    p_cha_wr : weight of ridge
    p_cha_thre : 25
    p_laMge_bottom : little area merge down
    p_laMge_top : little area merge up
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
        # pa.p_laMge_bottom = 64
        # pa.p_laMge_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.shape[0]*dataMgr.shape[1]/1600))
        # pa.p_gd_we = 1.0
        # pa.p_gd_wr = 1.2
        # pa.p_gd_thre = 20
        # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_laMge_bottom,p_laMge_top)
        # p_laMge_top = 400
        pass
    if DEF_LDATA:
        if not gray:
            pa.p_seg_color = 4
            pa.p_seg_dist = 0.5
            pa.p_cha_wc1 = 1
            pa.p_cha_wc23 = pa.p_cha_wc1*pa.p_seg_color
            pa.p_cha_we = 0.8
            pa.p_cha_wr = 1
            pa.p_cha_thre = 15
            pa.p_laMge_bottom = 64
            pa.p_laMge_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.shape[0]*dataMgr.shape[1]/1600))
            pa.p_gd_pow = 0.5
            pa.p_gd_we = 1.0
            pa.p_gd_wr = 1.2
            pa.p_gd_thre = 15
            # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_laMge_bottom,p_laMge_top)
        else:
            # print("gray")
            pa.p_seg_color = 0
            pa.p_seg_dist = 0.3
            pa.p_cha_wc1 = 1
            pa.p_cha_wc23 = pa.p_cha_wc1*pa.p_seg_color
            pa.p_cha_we = 2
            pa.p_cha_wr = 3
            pa.p_cha_thre = 7
            pa.p_laMge_bottom = 64
            pa.p_laMge_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.shape[0]*dataMgr.shape[1]/1600))
            pa.p_gd_pow = 0.5
            pa.p_gd_we = 1.0
            pa.p_gd_wr = 1.2
            pa.p_gd_thre = 7
            # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_laMge_bottom,p_laMge_top)
        
def processFile(data, test = False):
    dataMgr = DataMgr(data)
    
    # if not imglib.isGray(data):
    #     return 0

    # (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_laMge_bottom,p_laMge_top) = dataMgr.para
    param = dataMgr.para
    setPara(dataMgr,imglib.isGray(data), dataMgr.shape )
    start_time = time.time()
    getLargeSegment(dataMgr, 1000)
    output0 = dataMgr.region_copy()
    if np.mean(output0)!=np.mean(dataMgr.region.IntMatrix):
        print("Wrong!")
    print("RandomSeg edge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
    
    
    oc = []
    oce = []
    oc.append(dataMgr.data)
    oce.append(np.zeros(dataMgr.data.shape))

    # ------------------------------------- #
    # ---- mean merge                   --- #
    start_time = time.time()
    mergeThreshold = 2
    meanmergeRegionAdj(dataMgr, mergeThreshold)
    
    print("Mean merge while lamda=%d time:\t--- %8.4f seconds ---" % (mergeThreshold,time.time() - start_time))
    
    oc.append(toMean(dataMgr,False))
    oce.append(toMean(dataMgr,True))


    # # ------------------------------------- #
    # start_time = time.time()
    # mergeRegionBy2(dataMgr)
    # print("Original merge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
    # oc.append(toMean(dataMgr,False))
    # oce.append(toMean(dataMgr,True))
    if not test:
        # # ------------------------------------- #
        # # ---- little region delete         --- #
        # # p_cha_thre = 16
        # start_time = time.time()
        # mergeLittleRegion(dataMgr)
        # # out2 = dataMgr.region_copy()
        # print("Little region merge time:\t--- %8.4f seconds ---" % (time.time() - start_time))
        # oc.append(toMean(dataMgr,False))
        # oce.append(toMean(dataMgr,True))
        # ------------------------------------- #
        # ---- print chara 2                --- #
        start_time = time.time()
        mergeRegionContainArea(dataMgr)
        print("Area merge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True))
        # ------------------------------------- #
        # ---- merge mergeRegion_AG_3       --- #
        start_time = time.time()
        mergeRegion_AG_3(dataMgr)
        print("Area merge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True))
    else:

        # ------------------------------------- #
        # ---- print chara 2                --- #
        getChara2(dataMgr.Cdata, dataMgr.para)
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True))


        # ------------------------------------- #
        # ---- print chara 2                --- #
        reg1,reg2 = mergeRegion_AG_3(dataMgr)
        reg_f = imglib.addArrayCol(reg1,1)+imglib.addArrayCol(reg2,0)
        oc.append(imglib.charaToImg(reg_f,True))
        oce.append(toMean(dataMgr,True))

        # ------------------------------------- #
        # ---- mean merge                   --- #
        start_time = time.time()
        mergeThreshold = 2
        meanmergeRegionAdj(dataMgr, mergeThreshold)
        
        print("Mean merge while lamda=%d time:\t--- %8.4f seconds ---" % (mergeThreshold,time.time() - start_time))
        
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True))

        # ------------------------------------- #
        # ---- little region delete         --- #
        # p_cha_thre = 16
        start_time = time.time()
        mergeLittleRegion(dataMgr)
        # out2 = dataMgr.region_copy()
        print("Little region merge time:\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True))


        # ------------------------------------- #
        start_time = time.time()
        mergeRegionContainArea(dataMgr)
        # out2 = dataMgr.region_copy()
        print("Area merge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True))



    # ===================================== #
    outRow1 = imglib.mergeArray(tuple(oc),axis=1, interval=20)
    outRow2 = imglib.mergeArray(tuple(oce),axis=1, interval=20)
    # out2 = imglib.mergeArray(tuple(list2),axis=1, interval=20)
    
    
    del dataMgr
    return imglib.mergeArray((outRow1,outRow2),axis=0, interval=20)

