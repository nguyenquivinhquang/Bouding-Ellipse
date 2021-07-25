import math
import os
import cv2
from numpy.lib.function_base import angle
import torch
import numpy as np


def extract_ellipse_opencv(ellipse):
    x1,y1 = ellipse[0][0], ellipse[0][1]
    x2,y2 = ellipse[1][0], ellipse[1][1]
    angle = ellipse[2]
    return int(x1),int(y1),int(x2),int(y2),angle

DATA_FOLDER = 'D:\\AIC-Reid\AIC20_track2\\train_foreground\\'
img_path = DATA_FOLDER + '000009.jpg'

FILE_LABEL = './dataset/label.txt'
label = open(FILE_LABEL, 'a')

img = cv2.imread(img_path)
print(img.shape)
file_names = next(os.walk(DATA_FOLDER))[2]
total = 0
for file in file_names:
    try: 
        img = cv2.imread(DATA_FOLDER + file)

        imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgray = cv2.resize(imgray, (224,224))
        # cv2.imshow('a', imgray)
        ret,thresh = cv2.threshold(imgray,180,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnt = None
        area = 0
        for cn in contours:
            if cn.size < area: continue
            area = cn.size
            cnt = cn
        if area < 500: continue
        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)

        ellipse = cv2.fitEllipse(cnt)

        # ellipse[0][0] += 20
        
        x1,y1,len_a,len_b,_angle = extract_ellipse_opencv(ellipse)
        len_a += 20
        len_b += 20
        _angle = _angle * math.pi / 180.0
        ouput = '{file_name} {x} {y} {a} {b} {angle} \n'.format(file_name=file,x=x1,y=y1,a=len_a,b=len_b,angle=_angle)
        label.write(ouput)
        # cv2.waitKey(0)
        # break
    except: print("Error:", file)