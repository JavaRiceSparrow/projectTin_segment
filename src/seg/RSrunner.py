import numpy as np
from util import nodelib, imglib
from seg import seglib
# from seg.region import *
from seg.randomSeg import *
import time

in_path_str = "Pic/"

DEBUG = False
# TO_IMG = True

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

    if DEF_LDATA:
        pa.p_la_bottom =  int(max(dataMgr.getarea()/1600,100))
        pa.p_la_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.getarea()/400))
        if not gray:
            pa.p_seg_color = 4
            pa.p_seg_dist = 0.5
            pa.p_cha_wc1 = 1
            pa.p_cha_wc23 = pa.p_cha_wc1*pa.p_seg_color
            pa.p_cha_we = 2*6
            pa.p_cha_wr = 2*12
            pa.p_cha_thre = 15*5
            pa.p_gd_m = 10
            pa.p_gd_pow = 0.4
            pa.p_gd_we = 9*2*1.5
            pa.p_gd_wr = 12*2*1.5
            pa.p_gd_thre = 30*2
            pa.p_wth_amp = 2
            pa.p_wth_area  = 2
            pa.p_wth_grad  = 25
            pa.p_wth_thre  = 20*0.5
            # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top)
        else: #if gray
            # print("gray")
            pa.p_seg_color = 0
            pa.p_seg_dist = 0.14
            pa.p_cha_wc1 = 1
            pa.p_cha_wc23 = pa.p_cha_wc1*pa.p_seg_color
            pa.p_cha_we = 6*4
            pa.p_cha_wr = 10*6
            pa.p_cha_thre = 5*8
            # pa.p_la_bottom = int(max(16,dataMgr.getarea()/6400))
            # pa.p_la_top = int(max(400,max(dataMgr.shape[0],dataMgr.shape[1])/10.0*4,dataMgr.getarea()/1600))
            pa.p_gd_pow = 0.5
            pa.p_gd_m = 10
            pa.p_gd_we = 8*2*1.5
            pa.p_gd_wr = 12*2*1.5
            pa.p_gd_thre = 50*2
            pa.p_wth_amp = 4
            pa.p_wth_area  = 5
            pa.p_wth_grad  = 75
            pa.p_wth_thre  = 30*0.5
            # dataMgr.para = (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top)

def processFile(data, b_test = False,b_simp = False, b_print=True):
    dataMgr = DataMgr(data)
    
    # if not imglib.isGray(data):
    #     return 0

    # (p_seg_color,p_seg_dist,p_cha_wc1,p_cha_wc23,p_cha_we,p_cha_wr,p_cha_thre,p_la_bottom,p_la_top) = dataMgr.para
    param = dataMgr.para
    b_gray = imglib.isGray(data)
    setPara(dataMgr,imglib.isGray(data), dataMgr.shape )

    # if not b_gray:
    #     return None

    def myprint(str1):
        if b_print:
            print(str1)
    
    oc = []
    oce = []
    if b_simp:
        oce.append(dataMgr.data)
    else:
        oc.append(dataMgr.data)
        oce.append(dataMgr.Cdata)
    if not b_test:
        
        
        
        start_time = time.time()
        getLargeSegment(dataMgr, 2000,killTinyReg=False) ## TODO
        output0 = dataMgr.region_copy()
        # if np.mean(output0)!=np.mean(dataMgr.regMgr.space):
        #     print("Wrong!")
        
        myprint("RandomSeg edge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        # oc.append(toMean(dataMgr,False))
        # oce.append(toMean(dataMgr,True,b_gray))    

        # ---- little region delete         --- #
        # p_cha_thre = 16
        start_time = time.time()
        mergeLittleRegion(dataMgr)
        # out2 = dataMgr.region_copy()
        myprint("Little region merge time:\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))
        # ------------------------------------- #

        # # ------------------------------------- #
        # # ---- print chara 2                --- #
        # reg1,reg2 = getChara2(dataMgr.Cdata, dataMgr.para.get_cha())
        # reg_f = imglib.addArrayCol(reg1,0)+imglib.addArrayCol(reg2,1)
        # oc.append(imglib.charaToImg(reg_f,inv=True))
        # # oce.append(np.zeros(dataMgr.data.shape))
        # # ------------------------------------- #

        # ---- merge Chara2                 --- #
        start_time = time.time()
        mergeRegion_A_2(dataMgr)
        myprint("Merge2 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))        
        # ------------------------------------- #

        # ---- merge Chara2                 --- #
        # dataMgr.para.p_la_bottom = 
        dataMgr.para.p_la_top *= 8
        start_time = time.time()
        mergeRegion_A_2(dataMgr)
        myprint("Merge2-2 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))        
        # ------------------------------------- #

        # # ---- merge Chara31                --- #
        # start_time = time.time()
        # mergeRegion_AG_31(dataMgr)
        # print("Merge31 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        # oc.append(toMean(dataMgr,False))
        # oce.append(toMean(dataMgr,True,b_gray))
        # # ------------------------------------- #

        
    else:
        start_time = time.time()
        getLargeSegment(dataMgr, 2000,killTinyReg=False) ## TODO
        output0 = dataMgr.region_copy()
        # if np.mean(output0)!=np.mean(dataMgr.regMgr.space):
        #     print("Wrong!")
        myprint("RandomSeg edge time:\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        # oc.append(toMean(dataMgr,False))
        # oce.append(toMean(dataMgr,True,b_gray))    

        # ---- little region delete         --- #
        # p_cha_thre = 16
        start_time = time.time()
        mergeLittleRegion(dataMgr,dataMgr.para.p_la_bottom/32.0)
        # out2 = dataMgr.region_copy()
        myprint("Little region merge time:\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))
        # ------------------------------------- #


        # # ---- merge Chara2                 --- #
        # start_time = time.time()
        # mergeRegion_A_2(dataMgr)
        # myprint("Merge2 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        # # oc.append(toMean(dataMgr,False))
        # # oce.append(toMean(dataMgr,True,b_gray))        
        # # ------------------------------------- #

        # # ---- merge Chara2                 --- #
        # # dataMgr.para.p_la_bottom = 
        # dataMgr.para.p_la_top *= 4
        # start_time = time.time()
        # mergeRegion_A_2(dataMgr)
        # myprint("Merge2-2 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        # oc.append(toMean(dataMgr,False))
        # oce.append(toMean(dataMgr,True,b_gray))        
        # # ------------------------------------- #
        # dataMgr.reg2Prev(2)

        # ---- merge Chara3                 --- #
        start_time = time.time()
        mergeRegion_AG_3(dataMgr)
        myprint("Merge3 time:\t\t\t--- %8.4f seconds ---" % (time.time() - start_time))
        oc.append(toMean(dataMgr,False))
        oce.append(toMean(dataMgr,True,b_gray))
        # ------------------------------------- #

        # ===================================== #
    # ===================================== #
    # print(dataMgr.rmIdx)
    # print(len(dataMgr.rmList))
    # print(dataMgr.rlPrevList[1])
    print("Region num:\t\t\t--- %8d regions ---" %(dataMgr.regMgr.idxNum ))
    outRow2 = imglib.mergeArray(tuple(oce),axis=1, interval=20)
    if b_simp:
        return outRow2
    outRow1 = imglib.mergeArray(tuple(oc),axis=1, interval=20)
    # out2 = imglib.mergeArray(tuple(list2),axis=1, interval=20)
    
    
    del dataMgr
    return imglib.mergeArray((outRow1,outRow2),axis=0, interval=20)

    