import numpy as np
import os, sys
# from array import array

from util import imglib, color, nodelib
# import color
from seg import randomSeg, segment
# import nodelib

# path_name = ""

DEF_SAVEIMG = True ################################
DEF_BIGFILE_PASS = True ################################


list1 = ['004.BMP', '017.BMP', '019.BMP', '1.gif', '10.gif', '11.gif', '12.gif', '13.gif', '14.gif', '15.gif', '16.gif', '17.gif', '18.gif', '19.gif', '2.gif', '20.gif', '21.gif', '22.gif', '23.gif', '24.gif', '25.gif', '26.gif', '27.gif', '28.gif', '29.gif', '3.gif', '30.gif', '31.gif', '32.gif', '33.gif', '34.gif', '35.gif', '36.gif', '37.gif', '38.gif', '39.gif', '4.gif', '40.gif', '41.gif', '42.gif', '43.gif', '44.gif', '45.gif', '46.gif', '47.gif', '48.gif', '49.gif', '5.gif', '6.gif', '7.gif', '8.gif', '9.gif', '9mm_f3.2.JPG', '9mm_f8.JPG', 'BABOON - 複製.BMP', 'BABOON.BMP', 'Baboon1.bmp', 'Baboon3.bmp', 'Barbara.png', 'Barbara_color.jpg', 'Boats.png', 'DJJ.bmp', 'Datasource.txt', 'EARTH.BMP', 'FEFE.BMP', 'FRUIT.BMP', 'FreshFruitVegetable.jpg', 'FreshFruitVegetablegray.jpg', 'Heater2_gray.jpg', 'Heater_gray.jpg', 'House.png', 'IMG00g.bmp', 'Jellyfish.jpg', 'KEN.BMP', 'LAKE.BMP', 'LEAF.JPG', 'LEAF1.BMP', 'LEAF1.JPG', 'LENA.BMP', 'Lena - 複製.png', 'Lena.png', 'Lena1.bmp', 'Lena2.bmp', 'Lena256c.bmp', 'Lena256c.jpg', 'Lena512c.bmp', 'Lena_256bmp.bmp', 'Lena_gray_512.bmp', 'Lena_gray_label.bmp', 'PEPPER.BMP', 'Penguin.jpg', 'Pepper2c.bmp', 'Pepper512c.bmp', 'Peppers.png', 'SHUTTLE.BMP', 'TIFFANY.BMP', 'Yoyo_4month.JPG', 'airplane.bmp', 'apple.bmp', 'baboon2.jpg', 'baby.bmp', 'baby_test.bmp', 'balls.bmp', 'baseball.bmp', 'beach.jpg', 'bell-peppers-1420709__180.jpg', 'bowtest.bmp', 'car.jpg', 'coin.bmp', 'dato.JPG', 'datogray.jpg', 'desktop.ini', 'dofpro_toycarsDOF.jpg', 'door.jpg', 'drives.bmp', 'eclipse.bmp', 'flag.bmp', 'flag.psd', 'flag1.bmp', 'flag2.BMP', 'flag2.jpg', 'fruits.bmp', 'fruits1.bmp', 'graybaboon.bmp', 'house1.bmp', 'lena512.bmp', 'lena_128.bmp', 'lena_256.bmp', 'lena_gray.bmp', 'lena_gray_label.png', 'mandril.jpg', 'paddle.bmp', 'pens.bmp', 'pens_gray.bmp', 'pens_test.bmp', 'pepper1.bmp', 'peppers.bmp', 'peppers256.bmp', 'planets.bmp', 'planets_color.bmp', 'polygon.bmp', 'sailboat.jpg', 'splash.jpg', 'tea.bmp', 'tea.psd', 'tea1.bmp', 'tennis.bmp', 'tennis.psd', 'tennis_test.psd', 'test1.bmp', 'test2.bmp', 'test2.jpg', 'test2_1.bmp', 'test3.JPG', 'test3.bmp', 'test4.bmp', 'test5.bmp', 'test_a.bmp', 'test_pen.bmp', '海灘gray.jpg']

for path_name in list1:
    if path_name[-4:] == '.psd' or path_name[-4:] == '.ini' or path_name[-4:] == '.txt':
        list1.remove(path_name)

try:
    DEF_BIGFILE_PASS
except NameError:
    pass
else:
    list1.remove('Yoyo_4month.JPG')
    list1.remove('9mm_f8.JPG')
    list1.remove('9mm_f3.2.JPG')




import time

inpathHead = "Pic/"
outpathHead = "output/largeRSeg/2/"
# pathnames = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
pathnames = list1
# print(onlyfiles)

# if 
# fp = open("out1.txt", "w")



# '''
def testFile(path_name):
    
    in_path = inpathHead + path_name
    # out_path = outpathHead + path_name
    out_path = outpathHead + path_name + "_t1.gif"
    
    # lamda = 2
    data = imglib.getImg(in_path, to_3d=True)
    data_zero = data*0
    if type(data) == None:
        print("\""+in_path+"\" error!")
        return
    print("Image "+in_path+" ...")
    out = randomSeg.testFile(data)
    imglib.saveImg(out,out_path)
    
def processFile(path_name):
    in_path = inpathHead + path_name
    # out_path = outpathHead + path_name
    out_path = outpathHead + path_name + "_l1.gif"
    
    # lamda = 2
    data = imglib.getImg(in_path, to_3d=True)
    data_zero = data*0
    if type(data) == None:
        print("\""+in_path+"\" error!")
        return
    print("Image "+in_path+" ...")
    # start_time = time.time()
    out = randomSeg.processFile(data)
    imglib.saveImg(out,out_path)
    
# '''
def main(argv=None):
    if argv is None:
        argv = sys.argv
    argLen = len(argv)
    if argLen >= 2:
        if argv[1] == '-show':
            if argLen >= 3:
                for i in range(2,argLen):
                    n = int(argv[i] )
                    print("\"" + pathnames[n] + "\"")
            
            return 0
        if argv[1] == '-test':
            if argLen >= 3:
                for i in range(2,argLen):
                    n = int(argv[i] )
                    path_name = pathnames[n]
                    testFile(path_name)
            return 0
        
        elif argv[1] == '-a':
            if argLen == 3:
                n = int(argv[2] )
                for i in range(n,len(pathnames)):
                    processFile(pathnames[i])
            else:
                for path_name in pathnames:
                    processFile(path_name)
            return 0
        else:
            if argLen >= 2:
                for i in range(1,argLen):
                    n = int(argv[i] )
                    path_name = pathnames[n]
                    processFile(path_name)
            return 0

    else:
        for path_name in pathnames:
            processFile(path_name)
        return 0

if __name__ == "__main__":
    sys.exit(main())

# edge = nodelib.getEdge(array1)               

# x = 1
# for y in range(size_y-1):
#     if diff_right[x,y] <= 25:
#         seg_array[x,y+1] = seg_array[x,y]
#     else:
#         seg_array[x,y+1] = len(crs_arr)
#         crs_arr.append(seg_idx)
#         # crs_idx += 1
#         seg_idx += 1
    
# for x in range(1,size_x):
#     for y in range(size_y):
#         if diff_down[x-1,y] <= 25:
#             seg_array[x,y] = seg_array[x-1,y]

#     for y in range(size_y-1):
#         if diff_right[x,y] <= 25:
#             if seg_array[x,y+1] == 0:
#                 seg_array[x,y+1] = seg_array[x,y]
#             else: 
#                 n1,n2 = seg_array[x,y+1], seg_array[x,y]
#                 if crs_arr[n1] != crs_arr[n2]:
#                     crs_arr[n1] = crs_arr[n2]
#         elif seg_array[x,y+1]:
#             seg_array[x,y+1] = len(crs_arr)
#             crs_arr.append(seg_idx)
#             # crs_idx += 1
#             seg_idx += 1




