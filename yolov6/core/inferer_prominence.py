<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import time
import math
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
import copy


def padding(im, what, size):
    if what == 0:
        new_im = np.zeros((size, im.shape[1], im.shape[2]), im.dtype)
        new_im[int((size-im.shape[0])/2):int((size-im.shape[0])/2)+im.shape[0], :, :] = im[:, :, :]
    else:
        new_im = np.zeros((im.shape[0], size, im.shape[2]), im.dtype)
        new_im[:, int((size-im.shape[1])/2):int((size-im.shape[1])/2)+im.shape[1], :] = im[:, :, :]
    
    return new_im 


class Inferer:
    def __init__(self, weights, device, img_size):

        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        # Half precision
        if False & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup




    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")

    
    def inferv2(self, big_image, conf_thres, iou_thres, classes, agnostic_nms, max_det, conf_threshold, overlap_threshold, usemiddle=False):
        size = 640


        H = big_image.shape[0]
        W = big_image.shape[1]

        coordinates = []
        ind1 = 0
        ind2 = 0
        if H < size:
            big_image = padding(big_image, 0, size)
        if W < size:
            big_image = padding(big_image, 1, size)

        if usemiddle:
            width = []
            height = []
            while True:
                if ind1+size >= W:
                    width.append(W-size)
                    break
                else:
                    width.append(ind1)
                    ind1 += int(size*3/4)
            while True:
                if ind2+size > H:
                    height.append(H-size)
                    break                    
                else:
                    height.append(ind2)
                    ind2 += int(size*3/4)
            for x in width:
                coordinates.append([x, height[0]])
                coordinates.append([x, height[-1]])
            for y in height[1:-1]:
                coordinates.append([width[0], y])
                coordinates.append([width[-1], y])
            

        else:
            while True:
                temp = []
                
                
                if ind1+size >= W:
                    temp.append(W-size)
                    ind1 = -1
                else:
                    temp.append(ind1)
                    ind1 += int(size/2)
                if ind2+size > H:
                    temp.append(H-size)
                    ind2 = H-size
                    
                else:
                    temp.append(ind2)
                    if ind1 == -1:
                        if ind2 == H-size:
                            coordinates.append(temp)
                            break
                        ind2 += int(size/2)
                        ind1 = 0


                coordinates.append(temp)


        overlap_checker = np.zeros([H, W], dtype=np.uint8)
        returned = []

        for ind, (x, y) in enumerate(coordinates):
            image = np.zeros((size, size, 3), big_image.dtype)
            image = big_image[y:y+size, x:x+size]
            image = cv2.resize(image, (640, 640))
            duplicate = copy.deepcopy(image)

            #cv2.imwrite("b%d.png" % ind, duplicate)

            img, img_src = self.process_image(image, 640, self.stride, False)

            img = img.to(self.device)

            if len(img.shape) == 3:
                img = img[None]
            pred_results = self.model(img)

            
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            img_ori = img_src.copy()

            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            #self.font_check()

            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()


            for *xyxy, conf, cls in reversed(det):
                #if int(cls) != 0:
                #    continue
                
                #print(float(conf), conf_threshold)
                if float(conf) < conf_threshold:
                    continue

                x1 = int(xyxy[0]) + x
                y1 = int(xyxy[1]) + y
                x2 = int(xyxy[2]) + x
                y2 = int(xyxy[3]) + y

                block = overlap_checker[y1:y2, x1:x2].astype(np.float32)
                block_mean = np.mean(block)
                
                if block_mean > overlap_threshold:
                    continue
                
                overlap_checker[y1:y2, x1:x2] = 255
                new_xyxy = [x1, y1, x2, y2]
                line = [int(cls), new_xyxy, float(conf)] # cls will always be 1, meaining brittle stars because other objects were filtered out
                returned.append(line)

                cv2.rectangle(duplicate, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 3)
                

        overlap_checker = np.zeros([H, W], dtype=np.uint8)

        returned_boxes = []


        for n in returned[::-1]:
            
            x1, y1, x2, y2 = n[1]
            

            block = overlap_checker[y1:y2, x1:x2].astype(np.float32)
            block_mean = np.mean(block)
        
            if block_mean > overlap_threshold:
                continue
            overlap_checker[y1:y2, x1:x2] = 255
            returned_boxes.append(n)


        return returned_boxes
    

    @staticmethod
    def process_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
        return image

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color

class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
=======
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import time
import math
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
import copy


def padding(im, what, size):
    if what == 0:
        new_im = np.zeros((size, im.shape[1], im.shape[2]), im.dtype)
        new_im[int((size-im.shape[0])/2):int((size-im.shape[0])/2)+im.shape[0], :, :] = im[:, :, :]
    else:
        new_im = np.zeros((im.shape[0], size, im.shape[2]), im.dtype)
        new_im[:, int((size-im.shape[1])/2):int((size-im.shape[1])/2)+im.shape[1], :] = im[:, :, :]
    
    return new_im 


class Inferer:
    def __init__(self, weights, device, img_size):

        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        # Half precision
        if False & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup




    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")

    
    def inferv2(self, big_image, conf_thres, iou_thres, classes, agnostic_nms, max_det, conf_threshold, overlap_threshold, usemiddle=False):
        size = 640


        H = big_image.shape[0]
        W = big_image.shape[1]

        coordinates = []
        ind1 = 0
        ind2 = 0
        if H < size:
            big_image = padding(big_image, 0, size)
        if W < size:
            big_image = padding(big_image, 1, size)

        if usemiddle:
            width = []
            height = []
            while True:
                if ind1+size >= W:
                    width.append(W-size)
                    break
                else:
                    width.append(ind1)
                    ind1 += int(size*3/4)
            while True:
                if ind2+size > H:
                    height.append(H-size)
                    break                    
                else:
                    height.append(ind2)
                    ind2 += int(size*3/4)
            for x in width:
                coordinates.append([x, height[0]])
                coordinates.append([x, height[-1]])
            for y in height[1:-1]:
                coordinates.append([width[0], y])
                coordinates.append([width[-1], y])
            

        else:
            while True:
                temp = []
                
                
                if ind1+size >= W:
                    temp.append(W-size)
                    ind1 = -1
                else:
                    temp.append(ind1)
                    ind1 += int(size/2)
                if ind2+size > H:
                    temp.append(H-size)
                    ind2 = H-size
                    
                else:
                    temp.append(ind2)
                    if ind1 == -1:
                        if ind2 == H-size:
                            coordinates.append(temp)
                            break
                        ind2 += int(size/2)
                        ind1 = 0


                coordinates.append(temp)


        overlap_checker = np.zeros([H, W], dtype=np.uint8)
        returned = []

        for ind, (x, y) in enumerate(coordinates):
            image = np.zeros((size, size, 3), big_image.dtype)
            image = big_image[y:y+size, x:x+size]
            image = cv2.resize(image, (640, 640))
            duplicate = copy.deepcopy(image)

            #cv2.imwrite("b%d.png" % ind, duplicate)

            img, img_src = self.process_image(image, 640, self.stride, False)

            img = img.to(self.device)

            if len(img.shape) == 3:
                img = img[None]
            pred_results = self.model(img)

            
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            img_ori = img_src.copy()

            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            #self.font_check()

            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()


            for *xyxy, conf, cls in reversed(det):
                #if int(cls) != 0:
                #    continue
                
                #print(float(conf), conf_threshold)
                if float(conf) < conf_threshold:
                    continue

                x1 = int(xyxy[0]) + x
                y1 = int(xyxy[1]) + y
                x2 = int(xyxy[2]) + x
                y2 = int(xyxy[3]) + y

                block = overlap_checker[y1:y2, x1:x2].astype(np.float32)
                block_mean = np.mean(block)
                
                if block_mean > overlap_threshold:
                    continue
                
                overlap_checker[y1:y2, x1:x2] = 255
                new_xyxy = [x1, y1, x2, y2]
                line = [int(cls), new_xyxy, float(conf)] # cls will always be 1, meaining brittle stars because other objects were filtered out
                returned.append(line)

                cv2.rectangle(duplicate, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 3)
                

        overlap_checker = np.zeros([H, W], dtype=np.uint8)

        returned_boxes = []


        for n in returned[::-1]:
            
            x1, y1, x2, y2 = n[1]
            

            block = overlap_checker[y1:y2, x1:x2].astype(np.float32)
            block_mean = np.mean(block)
        
            if block_mean > overlap_threshold:
                continue
            overlap_checker[y1:y2, x1:x2] = 255
            returned_boxes.append(n)


        return returned_boxes
    

    @staticmethod
    def process_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
        return image

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color

class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
>>>>>>> master
