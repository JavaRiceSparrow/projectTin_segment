import numpy as np
import os, sys
# from array import array

import imglib
import color
from segment import *
import nodelib

path_name = ""

list1 = ['019.BMP', 'bowtest.bmp', 'lena_gray_label.png', 'Baboon3.bmp', '23.gif', '6.gif', '45.gif', '10.gif', 'paddle.bmp', 'lena512.bmp', '17.gif', '35.gif', 'Lena256c.bmp', 'Yoyo_4month.JPG', 'pens.bmp', '44.gif', 'fruits.bmp', 'TIFFANY.BMP', 'flag.psd', '28.gif', 'PEPPER.BMP', 'Barbara_color.jpg', '15.gif', 'drives.bmp', 'Lena_gray_512.bmp', 'baseball.bmp', 'tennis.psd', '22.gif', 'Boats.png', 'desktop.ini', 'Lena - 複製.png', 'balls.bmp', '7.gif', 'peppers256.bmp', 'planets.bmp', '37.gif', '5.gif', 'lena_256.bmp', '海灘gray.jpg', 'sailboat.jpg', 'Heater2_gray.jpg', 'beach.jpg', '4.gif', '9mm_f8.JPG', '017.BMP', 'flag.bmp', 'BABOON.BMP', 'FreshFruitVegetable.jpg', 'baby_test.bmp', 'fruits1.bmp', 'house1.bmp', 'test2.jpg', 'lena_128.bmp', '30.gif', 'Baboon1.bmp', '46.gif', 'pens_test.bmp', 'Lena_256bmp.bmp', 'flag2.jpg', '16.gif', 'door.jpg', 'LAKE.BMP', '43.gif', 'Barbara.png', 'DJJ.bmp', '40.gif', 'eclipse.bmp', '25.gif', '20.gif', 'FEFE.BMP', 'flag1.bmp', '9.gif', 'FRUIT.BMP', '41.gif', '2.gif', 'airplane.bmp', 'mandril.jpg', 'SHUTTLE.BMP', '1.gif', '48.gif', '12.gif', '47.gif', 'test2_1.bmp', '3.gif', 'baby.bmp', 'tea1.bmp', 'test4.bmp', 'bell-peppers-1420709__180.jpg', '9mm_f3.2.JPG', 'Penguin.jpg', '42.gif', 'splash.jpg', '39.gif', 'Lena256c.jpg', '8.gif', '29.gif', 'Jellyfish.jpg', 'LEAF1.BMP', '11.gif', 'IMG00g.bmp', '27.gif', 'Datasource.txt', 'BABOON - 複製.BMP', 'LEAF.JPG', 'test5.bmp', 'Heater_gray.jpg', 'baboon2.jpg', 'test2.bmp', '24.gif', 'graybaboon.bmp', '26.gif', '004.BMP', 'EARTH.BMP', 'Lena1.bmp', 'test_a.bmp', 'polygon.bmp', 'datogray.jpg', '38.gif', '13.gif', 'dofpro_toycarsDOF.jpg', 'car.jpg', '32.gif', 'LENA.BMP', 'Lena_gray_label.bmp', 'planets_color.bmp', '34.gif', 'pepper1.bmp', 'dato.JPG', 'Pepper2c.bmp', 'tennis_test.psd', 'coin.bmp', '31.gif', '14.gif', 'tea.bmp', 'test3.bmp', 'tennis.bmp', 'pens_gray.bmp', 'tea.psd', '19.gif', 'FreshFruitVegetablegray.jpg', '36.gif', 'Lena2.bmp', 'KEN.BMP', 'Lena512c.bmp', '33.gif', 'test_pen.bmp', 'House.png', 'Pepper512c.bmp', 'LEAF1.JPG', 'test3.JPG', 'lena_gray.bmp', '21.gif', 'Lena.png', '18.gif', 'test1.bmp', 'flag2.BMP', 'peppers.bmp', 'Peppers.png', 'apple.bmp', '49.gif']


# if len(sys.argv) != 2:
#     print("No parameter!")
#     os._exit(0)
#     # if ()

# path_name =  sys.argv[1] 


import time

in_path = "Pic/"
out_path = "output/"
pathnames = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
# print(onlyfiles)

def processFile(path_name):
    array0 = imglib.getImg(in_path+path_name, to_3d=True)
    # print("Shape: ", end="")
    # print(array0.shape)
    array1 = imglib.img3dTo2d(array0)

    start_time = time.time()
  
    seg_array = getSegment(array1, threshold=10)

    # if seg_array==0:
    #     os._exit(0)

    print("--- %s seconds ---" % (time.time() - start_time))

    # print(np.mean(seg_array))
    # array1[edge] = [255,255,255]
    size_x, size_y = array1.shape
    segment = np.zeros(array0.shape)
    print(segment.shape)
    for x in range(size_x):
        for y in range(size_y):
            # print(seg_array)
            if seg_array[x,y] !=0:
                if len(segment[x,y]) !=3:
                    print("?????")
                segment[x,y] = np.array(color.getHue(seg_array[x,y]*10))

    compare = imglib.mergeArray((array0, segment),axis=1, interval=20)
    imglib.saveImg(compare, out_path+path_name)

if len(sys.argv) == 2:
    if sys.argv[1] == '-a':
        for path_name in pathnames:
            processFile(path_name)

    i = int(sys.argv[1] )
    path_name = pathnames[i]
    processFile(path_name)

else:
    for path_name in pathnames:
        processFile(path_name)


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




