import cv2
import numpy as np
import math
from numpy.lib.type_check import imag 
import torch
from models.resnet import fbresnet18
import torchvision.transforms as transforms
from dataset.Custom_image_dataset import CustomImageDataset
from torchvision.io import read_image

#-------handle data ------- #
transform_train_list = [
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = transforms.Compose(transform_train_list)
def scale(ellipse):

    x,y,len_a, len_b, angle = ellipse[0]
    x = 224 * x + 112
    y = 224 * y + 112
    len_a = max(len_a * 224,0)
    len_b = max(len_b * 224,0)
    angle = abs(angle * 180)
    x,y,len_a,len_b = int(x),int(y),int(len_a),int(len_b)
    return x,y,len_a,len_b,angle

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
    _img = np.copy(img)
    print(_img.shape)
    cv2.ellipse(_img,((x,y),(len_a,len_b),angle),(0,255,0),2)
    save_path = './visualize/' + 'res_'+ filename
    cv2.imwrite(save_path, _img)
    return
if __name__ == "__main__":
    PATH ='Test_img/'
    WEIGHT_PATH = 'ellipse-epoch-117.pth'
    model = fbresnet18()
    checkpoint = torch.load(WEIGHT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    original = cv2.imread('Test_img\\000025.jpg',0)
    original = cv2.resize(original, (224,224))
    image = read_image('Test_img/000025.jpg') 
    print(type(image))
    image = data_transforms(image)
    c,h,w = image.shape
    image = image.reshape((1,c,h,w))
    
    result =  model(image)
    print(result)
    visualize_ellipse(original, result,'01.png')


