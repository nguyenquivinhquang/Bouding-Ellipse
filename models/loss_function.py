from typing import IO
import numpy
from math import *
import torch
from torch.nn import SmoothL1Loss
from torch import sqrt, cos, sin,square, max, min, abs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bouding_box(ellipse):
    """Find the minimum rectangle bouding ellipse

    Args:
    x (int): center of ellipse in x-coor    
    y (int): center of ellipse in y-coor 
    a (int): length of a
    b (int): length of b
    angel (int): rotate of ellipse

    Returns:
    x,y,delta_x, delta_y: the center of square and the length
    """
    total_batch, _ = ellipse.shape
    x,y,a,b,angel = ellipse[:,0],ellipse[:,1],ellipse[:,2],ellipse[:,3], ellipse[:,4]
    # print(x,y)
    delta_x = 2 * sqrt(square(a*cos(angel)) + square(b*sin(angel)))
    delta_y = 2 * sqrt(square(a*sin(angel)) + square(b*cos(angel)))
    delta_x /= 2
    delta_y /= 2
    
    result = torch.zeros((total_batch,4), device=device)

    result[:,0] = x-delta_x
    result[:,1] = y-delta_y

    result[:,2] = x+delta_x
    result[:,3] = y+delta_y
    return result

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # print(boxA[:,0])
    eps = 1e-8
    tensor_max = max(boxA, boxB)
    tensor_min = min(boxA, boxB)
    xA = tensor_max[:,0]
    yA = tensor_max[:,1]
    xB = tensor_min[:,2]
    yB = tensor_min[:,3]
    
    # compute the area of intersection rectangle
    interArea = max(xB - xA, torch.tensor([0.],device=device)) * max(yB - yA, torch.tensor([0.], device=device)) + eps
   
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[:,2] - boxA[:,0]) * (boxA[:,3] - boxA[:,1]))
    boxBArea = abs((boxB[:,2] - boxB[:,0]) * (boxB[:,3] - boxB[:,1]))
    # print(boxAArea,boxBArea)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
def euclidean_distance(x1,y1, x2,y2):
   
    d = torch.pow(x1 -y1,2) + torch.pow(x2 -y2,2)
    return torch.pow(d, float(1/2))

def get_IOU_loss(outputs, targets):
    box_outputs = get_bouding_box(outputs)
    box_targets = get_bouding_box(targets)
    IOU =1 - bb_intersection_over_union(box_outputs, box_targets)
    return IOU
class ellipse_loss(object):
    def __init__(self):
        self.smooth_L1 = SmoothL1Loss()

    def __call__(self, outputs, targets):
        # print(outputs.shape)
       
        
        area_loss =get_IOU_loss(outputs,targets)
        center_loss = self.smooth_L1(outputs[:,0:1], targets[:,0:1])
        angle_loss = self.smooth_L1(abs(outputs[:,4]) , abs(targets[:,4]))
       
        return angle_loss + area_loss + center_loss


if __name__ == "__main__":
    # print(get_bouding_box(7,7,5,3,pi))
    # print(euclidean_distance([3,3],[5,5]))
    # print(atan2(1, 0))
    # bbA = torch.randint(10, (2, 4))
    # bbB = torch.randint(10, (2, 4))
    bbA = torch.tensor([[3,3,6,6]])
    bbB = torch.tensor([[3,5,6,6]])
    print(bbA, '\n-----------\n', bbB)
    print(bb_intersection_over_union(bbA, bbB))