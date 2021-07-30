import math
from typing import IO
import numpy
from math import *
import torch
from torch.nn import SmoothL1Loss
from torch import sqrt, cos, sin,square, max, min, abs,atan2

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def smooth_l1_loss(input, target = None, beta = 1.0,reduction='none'):
    if target == None: target = torch.zeros_like(input)
    if beta < 1e-3:
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

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
    IOU = 1 - bb_intersection_over_union(box_outputs, box_targets)
    return IOU

def compute_diff_angle(outputs, targets):

    return abs(abs(outputs[:,4]) - abs(targets[:,4]))

def scale_angle(outputs):
    # cos_angle = torch.abs(cos(angle))
    # cond = cos_angle >= 0
    # sin_angle = torch.where(cond, sin(angle), -sin(angle))

    # atan2_angle = atan2(sin_angle, cos_angle) / pi
    outputs = outputs + 1
    outputs = torch.remainder(outputs, 1)

    return outputs


class ellipse_loss(object):
    def __init__(self):
        self.smooth_L1 = SmoothL1Loss(reduction='mean')

    def __call__(self, outputs, targets):
        # print(outputs.shape)
       
        
        # area_loss = abs((get_IOU_loss(outputs,targets))).mean()
        center_loss = self.smooth_L1(outputs[:, 0:4], targets[:,0:4])

        angle_loss = self.smooth_L1(scale_angle(outputs[:,4]), scale_angle(targets[:,4]))

        # print(area_loss, center_loss, angle_loss)
        # print(angle_loss  + center_loss + area_loss)
        return angle_loss  + center_loss


if __name__ == "__main__":
    # print(get_bouding_box(7,7,5,3,pi))
    # print(euclidean_distance([3,3],[5,5]))
    # print(atan2(1, 0))
    # bbA = torch.randint(10, (2, 4))
    # bbB = torch.randint(10, (2, 4))
    # bbA = torch.tensor([[3,3,6,6,5]])
    # bbB = torch.tensor([[3,5,6,6,6]])
    # print(bbA, '\n-----------\n', bbB)
    # # print(bb_intersection_over_union(bbA, bbB))
    # print(bbA[:,4])
    alpha = pi / 4
    bbA = torch.rand((2, 1))
    bbA[0] = -1/3
    bbA[1] = 4/3
    print(scale_angle(bbA))
    