# from dataset.Custom_image_dataset import CustomImageDataset
import sys
import torch
sys.path.append("D:\python\Bouding-Ellipse")


import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import config.Default as cf
from torchvision.io import read_image
import os
import json
from torch.autograd import Variable
from PIL import Image

class_idx = json.load(open("imagenet_class_index.json"))

resnext50_32x4d = models.resnext50_32x4d(pretrained=True)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

path = cf.DATASET_PATH + '\\image_train\\'
f = open("label.txt", 'w')
centre_crop = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for file in os.listdir(path):
    file_img = path + file
    # img = read_image(file_img)
    # img = img.unsqueeze_(-1)
    # img = img.transpose(3, 0)
    # img = img.transpose(3, 1)
    img = Image.open(file_img)

    # print(type(img))
    # img = img.float()
    # result = resnext50_32x4d(img)
    # idx = int(torch.argmax(result))
    # out = resnext50_32x4d(Variable(centre_crop(img).unsqueeze(0)))
    # idx = int(torch.argmax(out))
    # # print(idx)
    # # print(class_idx[str(idx)])
    # f.write(file + ' ' + class_idx[str(idx)][1] +'\n')
    # print(file)