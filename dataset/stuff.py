# from dataset.Custom_image_dataset import CustomImageDataset
import sys
from numpy.lib.function_base import copy
import torch
import pandas as pd

import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from torchvision.io import read_image
import os
from torch.autograd import Variable
from PIL import Image
import shutil

def copy_img():
    df =pd.read_csv('dataset/train_label.csv')
    files = df['Image_name']
    total = 0
    print(files.shape)
    for file in files:
        src = '/mnt/Data/AIC-Reid/AIC20_track2/AIC20_ReID/image_train/' + file
        dst = 'dataset/img/'
        shutil.copy(src,dst)
        total += 1
    print(total)
    return

if __name__ == "__main__":
    copy_img()