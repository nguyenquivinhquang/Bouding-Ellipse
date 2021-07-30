import math
import os
import cv2
from numpy.lib.function_base import angle
import torch
import numpy as np
import csv
from pandas import DataFrame
def extract_ellipse_opencv(ellipse):
    x1,y1 = ellipse[0][0], ellipse[0][1]
    x2,y2 = ellipse[1][0], ellipse[1][1]
    angle = ellipse[2]
    return int(x1),int(y1),int(x2),int(y2),angle
def visualize_ellipse(img, ellipse, filename):
    """Visuzelize ellipse to img and save

    Args:
        img ([type]): [description]
        ellipse: ((x,y), (a,b), angle)
        filename filename 
    """
    _img = np.copy(img)
    cv2.ellipse(_img,ellipse,(0,255,0),2)
    save_path = './visualize/' + 'ell_'+ filename
    cv2.imwrite(save_path, _img)
    return


def extract_ellipse(DATA_FOLDER, FILE_LABEL, visualize = False ):

    file_names = next(os.walk(DATA_FOLDER))[2]
    total = 0
    table = []
    for file in file_names:
        try: 
            img = cv2.imread(DATA_FOLDER + file)
            img = cv2.resize(img, (224,224))

            imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
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
            # x,y,w,h = cv2.boundingRect(cnt)
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
            ellipse = cv2.fitEllipse(cnt)

            # ellipse[0][0] += 20
            

            x1,y1,len_a,len_b,_angle = extract_ellipse_opencv(ellipse)
            
            len_a += 20
            len_b += 20
            if visualize == True: visualize_ellipse(img, ((x1,y1), (len_a, len_b), _angle), file)
            # _angle = _angle * math.pi / 180.0
            ouput = '{file_name} {x} {y} {a} {b} {angle} {_a} \n'.format(file_name=file,x=x1,y=y1,a=len_a,b=len_b,angle=_angle,_a = _angle * math.pi / 180.0)
            total +=1 
            print(_angle,  _angle * math.pi / 180.0)
            # table.append([file,x1,y1,len_a, len_b, _angle * math.pi / 180.0])
            # if (total > 100): break
            # label.write(ouput)
            # print('Ok:', file)
            # cv2.waitKey(0)
            # break
        except: print("Error:", file)
    # df = DataFrame(table, columns=['Image_name', 'center_x','center_y','len_a','len_b','angle_in_radian'])
    # df.to_csv(FILE_LABEL, header=True, index=False)
        
    return


if __name__ == "__main__":

    DATA_FOLDER = 'D:/AIC-Reid/AIC20_track2/train_foreground/'
    FILE_LABEL = './dataset/train_label.csv'

    extract_ellipse(DATA_FOLDER=DATA_FOLDER, FILE_LABEL=FILE_LABEL, visualize=False)