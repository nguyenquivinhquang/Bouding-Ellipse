import cv2
import numpy as np
import math
from numpy.lib.type_check import imag 
import torch
from models.resnet import fbresnet18
import torchvision.transforms as transforms
from dataset.Custom_image_dataset import CustomImageDataset
from torchvision.io import read_image
from math import cos, sin
import os
#-------handle data ------- #
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = transforms.Compose(transform_train_list)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def scale(ellipse):

    x,y,len_a, len_b, angle = ellipse[0]
    x = 224 * x + 112
    y = 224 * y + 112
    len_a = max(len_a * 224,0)
    len_b = max(len_b * 224,0)
    angle = abs(angle * 180)
    x,y,len_a,len_b = int(x),int(y),int(len_a),int(len_b)
    return x,y,len_a,len_b,angle
def bouding_rectangle(x,y,len_a,len_b,angle, img):
    l_a = len_a / 2
    l_b = len_b / 2
    angle = angle * math.pi / 180
    x1 = int(x + l_b * cos(angle))
    y1 = int(y + l_b * sin(angle))
    x2 = int(x - l_b * cos(angle))
    y2 = int(y - l_b * sin(angle))
    # print(x1,y1,x2,y2,cos(angle), sin(angle))
    img = cv2.line(img, (x1,y1),(x2,y2), (0, 155, 0), 2)
   
    angle -= math.pi / 2
    x1 = int(x - l_a * cos(angle))
    y1 = int(y + l_a * sin(angle))
    x2 = int(x + l_a * cos(angle))
    y2 = int(y - l_a * sin(angle))
    # print(x1,y1,x2,y2,cos(angle), sin(angle))
    img = cv2.line(img, (x1,y1),(x2,y2), (0, 155, 0), 2)
    return
def visualize_ellipse(img, ellipse, filename):
    """Visuzelize ellipse to img and save

    Args:
        img ([type]): [description]
        ellipse: ((x,y), (a,b), angle)
        filename filename 
    """
    try: ellipse = ellipse.detach().numpy()
    except: pass
    x,y,len_a,len_b,angle = scale(ellipse)
    print(x,y,len_a,len_b,angle)
    _img = np.copy(img)
    cv2.ellipse(_img,((x,y),(len_a,len_b),angle),(0,255,0),2)
    # bouding_rectangle( x,y,len_a,len_b,angle, _img)
    save_path = './visualize/' + 'res_'+ filename
    cv2.imwrite(save_path, _img)
    cv2.waitKey(0)
    return


if __name__ == "__main__":
    PATH ='D:/AIC-Reid/AIC20_track2/AIC20_ReID/image_test/'
    WEIGHT_PATH = 'checkpoint/ellipse-epoch-298.pth'
    model = fbresnet18()
    checkpoint = torch.load(WEIGHT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    files = os.listdir(PATH)
    for file in files:
        original = cv2.imread(PATH + file)
        original = cv2.resize(original, (224,224))
        image = read_image(PATH + file) 

        image = data_transforms(image)
        c,h,w = image.shape
        image = image.reshape((1,c,h,w))
        # image = image.to(device)
        result =  model(image)
        visualize_ellipse(original, result,file)


