import numpy
from math import *
def get_bouding_box(x,y,a,b,angel):
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
    
    delta_x = 2 * sqrt(a**2 * cos(angel) ** 2 + b ** 2 * sin(angel)**2)
    delta_y = 2 * sqrt(a**2 * sin(angel) ** 2 + b ** 2 * cos(angel)**2)
    return x,y,delta_x,delta_y

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def ellipse_loss(outputs, targets):

    return
if __name__ == "__main__":
    print(get_bouding_box(1,1,4,1,pi))
