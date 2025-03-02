


#################################################################################################
#
#  Additional functions for program
#  Functions are used for both UI and batch-running
#  Uses code to run algorithm from yolov6/core/inferer_prominence.py
#
#################################################################################################


import cv2
import numpy as np
import math
import sys
import os
import pandas as pd
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import warnings
warnings.filterwarnings("ignore")
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def rotate_image(image, angle):
    '''
    rotates given image by selected angle
    '''
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def to_rgb(im):
    '''
    changes image from bgr to rgb
    '''
    h, w = im.shape
    ret = np.zeros((h, w, 3), dtype=np.uint8)
    ret[:, :, 0] = im * 255
    ret[:, :, 1] = im * 255
    ret[:, :, 2] = im * 255
    return ret

def polar(x, y, centerx, centery):
    '''
    converts polar to cartasian coords
    '''
    phi = np.arctan2(y-centery, x-centerx)
    polar_degrees = (phi*180.0)/math.pi
    return polar_degrees

def xy(phi, rho, centerx, centery):
    '''
    converts cartasian to polar coords
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return int(x+centerx+0.5), int(y+centery+0.5)

def image_height(box):
    '''
    finds the very top of the prominence
    '''
    for n in range(0, box.shape[0]):
        if np.mean(box[n,:,:]) > 60:
            return n

def line_height(box):
    '''
    finds base of the prominence
    '''
    for n in range(0, box.shape[0]):
        if box[n, box.shape[1]//2, 0] == 255 and box[n, box.shape[1]//2, 1] == 0 and box[n, box.shape[1]//2, 2] == 255:
            return n

def prom_area(box, lineheight):
    '''
    calculates area of prominence
    '''
    area = 0
    for n in range(0, lineheight):
        for m in range(0, box.shape[1]):
            if np.mean(box[n,m,:]) > 50:
                area += 1
    return area


def radian(degree):
    '''
    converts degrees to radians
    '''
    return (degree/180)*math.pi

def get_stat(img):
    '''
    finds average darkest and brightest spots by examining the four corners and center of the image
    '''
    mean1 = np.mean(img[:100, :100])
    mean2 = np.mean(img[-100:, -100:])
    mean3 = np.mean(img[:100, -100:])
    mean4 = np.mean(img[-100:, :100])
    dark = (mean1+mean2+mean3+mean4)/4.0
    bright = np.mean(img[img.shape[0]//2-50:img.shape[0]//2+50, img.shape[1]//2-50:img.shape[1]//2+50])
    return dark, bright


def draw_ellipse(degree1, degree2, sun_radius, sun_cx, sun_cy, height, duplicate, color=False):
    
    '''
    draws on ellipses onto images
 
    @param degree1: left hand degree of ellipse
    @param degree2: right hand degree of ellipse
    @param sun_radius: radius of the sun
    @param sun_cx: center coord of the sun (x)
    @param sun_cy: center coord of the sun (y)
    @param height: height of the prominence
    @param duplicate: the image
    @param color: (ONLY USED IN UI) if set to true, drawn ellipse will be yellow
    '''

    for deg in range(int(degree1*100), int(degree2*100)):
        fx1, fy1 = xy(radian(deg/100), sun_radius, sun_cx, sun_cy)
        if color == True:
            cv2.circle(duplicate, (fx1, fy1), 1,(255, 255, 0), 3)
        else:
            cv2.circle(duplicate, (fx1, fy1), 1,(0, 255, 0), 1)

        fx1, fy1 = xy(radian(deg/100), sun_radius+height, sun_cx, sun_cy)
        if color == True:
            cv2.circle(duplicate, (fx1, fy1), 1,(255, 255, 0), 3)
        else:
            cv2.circle(duplicate, (fx1, fy1), 1,(0, 255, 0), 1)

def format_boxes(boxes, sun_radius, sun_cx, sun_cy, duplicate2, show_boxes=False, img_dir="", duplicate=None): 
    '''
    converts rectangular boxes that are obtained from yolo algorithm into elliptical boxes using polar coordinates
      
    @param boxes: rectangular boxes outputed from yolo algorithm
    @param sun_radius: radius of ellipse
    @param sun_cx and sun_cy: center coordinates of ellipse
    @param duplicate2: sun image
    @param show_boxes: determines whether boxes will be drawn onto the image
    @param img_dir: directory that image will be saved in if csv is used to save file
    @param duplicate: sun image that will not be tampered throughout the process
     
    @return saved_bboxes: data of prominence that can be inputed into chart or saved in csv file
    @return polar_coords: (ONLY FOR UI) data used to redraw elliptical boxes when confidence is changed
    @return duplicate: image with annotated boxes
    '''
    polar_coords = []
    saved_bboxes = []
    for index, line in enumerate(boxes):
        x1, y1, x2, y2 = line[1]
        if int(line[0]) == 1:
            midx = (x1+x2)/2
            midy = (y1+y2)/2
            circumference = sun_radius*2.0*math.pi
            distance = max(abs(x2-x1), abs(y2-y1))
            angular_distance = (distance*360.0)/circumference
            degrees = polar(midx, midy, sun_cx, sun_cy)
            box = rotate_image(duplicate2[y1:y2, x1:x2, :], degrees+90)
            try:
                lineheight = line_height(box) # distance from top of box to line
                extra = abs(image_height(box)-lineheight)
                box2 = box[0:lineheight, :, :]
                area = prom_area(box, lineheight)
                #(str(index)+' '+str(area)+".png", box)
                #cv2.imwrite(str(index)+' '+str(area)+"top.png", box2)
                degree1 = degrees-(angular_distance/2.0)
                degree2 = degrees+(angular_distance/2.0)

                px1, py1 = xy(radian(degree1), sun_radius, sun_cx, sun_cy)
                px2, py2 = xy(radian(degree1), sun_radius+extra, sun_cx, sun_cy)
                px3, py3 = xy(radian(degree2), sun_radius, sun_cx, sun_cy)
                px4, py4 = xy(radian(degree2), sun_radius+extra, sun_cx, sun_cy)

                temp = [float(line[2]), px1, py1, px2, py2, px3, py3, px4, py4 ,degree1, degree2, extra, area]
                polar_coords.append(temp)
                if show_boxes == True:
                    draw_ellipse(degree1, degree2, sun_radius, sun_cx, sun_cy, extra, duplicate)
                    cv2.line(duplicate, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.line(duplicate, (px3, py3), (px4, py4), (0, 255, 0), 2)
                temp2 = [img_dir, str(degree1), str(degree2), str(extra), str(float(line[2])), str(area), sun_cx, sun_cy, sun_radius]
                saved_bboxes.append(temp2)
            except:
                continue

    return saved_bboxes, polar_coords, duplicate
    
def save_files(saved_bboxes, filepath):
    '''
    saves bboxes in csv file
    '''
    my_df = pd.DataFrame(saved_bboxes, columns=['Image', 'Left Edge', 'Right Edge', 'Height', 'Confidence', "Area", "CenterX", "CenterY", "Radius"])
    my_df.to_csv(filepath, index=False)

