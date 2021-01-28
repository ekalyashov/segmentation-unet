import cv2
import os
import numpy as np
from PIL import Image

from collections import OrderedDict

import data.dataset_seg as ds

class Callback:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

color_encoding = OrderedDict([
    ('sky', (128, 128, 128)),
    ('building', (128, 0, 0)),
    ('pole', (192, 192, 128)),
    ('road_marking', (128, 0, 192)),
    ('road', (128, 64, 128)),
    ('pavement', (0, 0, 192)),
    ('tree', (128, 128, 0)),
    ('sign_symbol', (192, 128, 128)),
    ('fence', (64, 64, 128)),
    ('car', (64, 0, 128)),
    ('pedestrian', (64, 64, 0)),
    ('bicyclist', (0, 128, 192)),
    ('unlabeled', (0, 0, 0)),

    ('truck', (64, 128, 192)),
    ('bus', (192, 128, 192)),
    ('text', (128, 128, 64)),
    ('traffic_light', (0, 64, 64)),
    ('wall', (64, 192, 0)),
    ('vegetation_misc', (192, 192, 0))
])

        
class DiceCoefCamVidCallback(Callback):
    def __init__(self, mask_path, origin_img_size, threshold):
        """
            Calculates statistic of predictions: 
                mean dice coefficient for every category of predicted objects, 
                enumerated in 'color_encoding' dictionary
            Args:
                mask_path (string): path to ground truth mask files
                origin_img_size (tuple): original size of mask images
                threshold (float): The threshold used to consider the mask present or not
            
        """
        self.threshold = threshold
        self.origin_img_size = origin_img_size
        self.mask_path = mask_path
        self.num_channels = 13
        self.color2idx = []
        self.diceDict = OrderedDict()
        self.colorKeys = list(color_encoding.keys())
        self.counter = 0
        for i in range(len(self.colorKeys)):
            self.diceDict[self.colorKeys[i]] = [0, 0.0]
        car_idx = 9
        sign_idx = 7
        fence_idx = 8
        tree_idx=6
        for index, (class_name, color) in enumerate(color_encoding.items()):
            if (class_name == 'car'):
                car_idx = index
            elif (class_name == 'sign_symbol'):
                sign_idx = index
            elif (class_name == 'fence'):
                fence_idx = index
            elif (class_name == 'tree'):
                tree_idx = index
            if (class_name == 'truck' or class_name == 'bus'):
                self.color2idx.append((color, car_idx))
            elif (class_name == 'text' or class_name == 'traffic_light'):
                self.color2idx.append((color, sign_idx))
            elif (class_name == 'wall'):
                self.color2idx.append((color, fence_idx))
            elif (class_name == 'vegetation_misc'):
                self.color2idx.append((color, tree_idx))
            else:
                self.color2idx.append((color, index))

    def dice_coef(self, y_true, y_pred):
        smooth = 0.000001
        y_true_f = np.ndarray.flatten(y_true)
        y_pred_f = np.ndarray.flatten(y_pred)
        intersection = np.sum(y_true_f * y_pred_f)
        return (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

    def __call__(self, *args, **kwargs):
        """
            Performs the calculation of dice coefficients,
                temporary accumulate them for final calculations.
        """
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']
        # compare predictions with true masks
        for (pred, name) in zip(probs, files_name):
            mask = pred > self.threshold
            
            split = os.path.splitext(name)
            base_name = split[0]
            true_mask = self.readMask(self.mask_path + '/' + base_name + '_L.png')
            for i in range(self.num_channels):
                m = mask[i]
                t_m = true_mask[i]
                if (np.sum(t_m) == 0):
                    continue
                dice_c = self.dice_coef(t_m, m)
                key = self.colorKeys[i]
                e = self.diceDict[key]
                e[1] = e[1] + dice_c
                e[0] = e[0] + 1
                self.diceDict[key] = e
            self.counter += 1
            
    def readMask(self, fName):
        img = Image.open(fName)
        i_width, i_height = img.size
        
        img = np.asarray(img.convert("RGB"), dtype=np.uint8)
        img = np.transpose(img, (2, 0, 1))
        mask = np.zeros((self.num_channels, i_height, i_width), dtype=np.float32)
        for e in self.color2idx:
            if (self.num_channels <= e[1]):
                break
            color = e[0]
            m_idx = np.logical_and(np.logical_and(img[0] == color[0], img[1] == color[1]), img[2] == color[2])
            mask[e[1], m_idx] = 1.
        
        return mask   
        
    def print_res(self):
        """
            Method to print calculated statistic, 
            should be called after prediction completion.
        """
        for key in self.diceDict.keys():
            e = self.diceDict[key]
            mean = 0. #mean value if class presented on image
            if (e[0] > 0):
                mean = e[1] / e[0]
            print("%s  %.4f (%d)" % (key, mean, e[0]))
        

class PredictionsCamVidSaverCallback(Callback):
    def __init__(self, in_path, out_path, maskOnly=False):
        """
            Stores prediction tensor as image, combining with input image.
            Args:
                in_path (string): path to folder where test images placed
                out_path (string): path to folder where predicted images should be stored
                maskOnly (bool): if true, only prediction tensor stores,
                    else predicted image and input image combines into one result
                
        """
        self.out_path = out_path
        self.in_path = in_path
        self.maskOnly = maskOnly
        self.color2idx = []
        for index, (class_name, color) in enumerate(ds.camvid_color_encoding.items()):
            self.color2idx.append((color, index))
        
#    color_encoding = OrderedDict([
#        ('sky', (128, 128, 128)),
#        ('building', (128, 0, 0)),
#        ('pole', (192, 192, 128)),
#        ('road_marking', (255, 69, 0)),
#        ('road', (128, 64, 128)),
#        ('pavement', (60, 40, 222)),
#        ('tree', (128, 128, 0)),
#        ('sign_symbol', (192, 128, 128)),
#        ('fence', (64, 64, 128)),
#        ('car', (64, 0, 128)),
#        ('pedestrian', (64, 64, 0)),
#        ('bicyclist', (0, 128, 192)),
#        ('unlabeled', (0, 0, 0))])
        
    def __call__(self, *args, **kwargs):
        """
            Performs the conversion from predicted tensor to a mask image,
                colored according to 'color2idx' map.
            Combines mask with input image and stores result to file.
        """
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']
        # Save the predictions
        for (pred, name) in zip(probs, files_name):

            mask = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
            for e in self.color2idx:
                if (pred.shape[0] <= e[1]):
                    break
                color = e[0]
                m_idx = pred[e[1]] > 0.5
                mask[m_idx] = color
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            if (self.maskOnly):
                cv2.imwrite(self.out_path + '/m_' + name + '.png' , mask)
            else:
                image = cv2.imread(self.in_path + '/' + name) 
                masked_image = cv2.addWeighted(mask, 0.6, image, 0.4, 0.)  # image * α + mask * β + λ
                cv2.imwrite(self.out_path + '/m_' + name + '.png', masked_image)
        
class PredictionsKittiSaverCallback(Callback):
    def __init__(self, in_path, out_path, maskOnly=False):
        """
            Stores prediction tensor as image, combining with input image.
            Args:
                in_path (string): path to folder where test images placed
                out_path (string): path to folder where predicted images should be stored
                maskOnly (bool): if true, only prediction tensor stores,
                    else predicted image and input image combines into one result
                
        """
        self.out_path = out_path
        self.in_path = in_path
        self.maskOnly = maskOnly
        self.color2idx = []
        for index, (class_name, color) in enumerate(ds.camvid_color_encoding.items()):
            self.color2idx.append((color, index))
        
    def __call__(self, *args, **kwargs):
        """
            Performs the conversion from predicted tensor to a mask image,
                colored according to 'color2idx' map.
            Combines mask with input image and stores result to file.
        """
        if kwargs['step_name'] != "predict":
            return

        probs = kwargs['probs']
        files_name = kwargs['files_name']
        # Save the predictions
        for (pred, name) in zip(probs, files_name):

            mask = np.zeros((pred.shape[1], pred.shape[2], 3), dtype=np.uint8)
            for e in self.color2idx:
                if (pred.shape[0] <= e[1]):
                    break
                color = e[0]
                m_idx = pred[e[1]] > 0.5
                mask[m_idx] = color
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
            if (self.maskOnly):
                cv2.imwrite(self.out_path + '/m_' + name + '.png' , mask)
            else:
                image = cv2.imread(self.in_path + '/' + name) 
                masked_image = cv2.addWeighted(mask, 0.6, image, 0.4, 0.)  # image * α + mask * β + λ
                cv2.imwrite(self.out_path + '/m_' + name + '.png', masked_image)