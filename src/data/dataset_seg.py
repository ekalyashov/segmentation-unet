#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:07:35 2018

@author: kev
"""
import numpy as np
from PIL import Image
import cv2

import torch.utils.data as data
import img.transformer as transformer

from collections import OrderedDict

# Default encoding for pixel value, class name, and class color
camvid_color_encoding = OrderedDict([
    ('sky', (128, 128, 128)),           #0
    ('building', (128, 0, 0)),          #1
    ('pole', (192, 192, 128)),          #2
    ('road_marking', (128, 0, 192)),    #3
    ('road', (128, 64, 128)),           #4
    ('pavement', (0, 0, 192)),          #5
    ('tree', (128, 128, 0)),            #6
    ('sign_symbol', (192, 128, 128)),   #7
    ('fence', (64, 64, 128)),           #8
    ('car', (64, 0, 128)),              #9
    ('pedestrian', (64, 64, 0)),        #10
    ('bicyclist', (0, 128, 192)),       #11
    ('unlabeled', (0, 0, 0)),           #12

    ('truck', (64, 128, 192)),          #9
    ('bus', (192, 128, 192)),           #9
    ('text', (128, 128, 64)),           #7
    ('traffic_light', (0, 64, 64)),     #7
    ('wall', (64, 192, 0)),             #8
    ('vegetation_misc', (192, 192, 0))  #6
])
  
kitti_idx_encoding = OrderedDict([      
    (0, [23]),                      #sky
    (1, [11]),                      #building
    (2, [17]),                      #pole
    (3, [255]),                     #road_marking
    (4, [7]),                       #road
    (5, [8]),                       #pavement, sidewalk
    (6, [21, 22]),                  #tree
    (7, [19, 20]),                  #sign_symbol
    (8, [12, 13]),                  #fence
    (9, [26, 27, 28, 29, 30, 31]),  #car
    (10, [24, 25]),                 #pedestrian
    (11, [32, 33]),                 #bicyclist
    (12, [0,1,2,3,4,5])             #unlabeled
])
        
class TrainCamVidDataset(data.Dataset):
    def __init__(self, X_data, y_data, img_resize,
                 X_transform=None, threshold=0.5, multiplier = None):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()

            Args:
                X_data (list): List of paths to the training images
                y_data (list, optional): List of paths to the target images
                img_resize (tuple): Tuple containing the new size of the images
                X_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                threshold (float): The threshold used to consider the mask present or not
                multiplier (int, optional): increases dataset size by multiplier, 
                    data lists used cyclically
        """
        self.threshold = threshold
        self.X_train = X_data
        self.y_train_masks = y_data
        self.X_transform = X_transform
        self.multiplier = multiplier
        self.maxLen = len(self.X_train)
        self.img_resize = img_resize
        self.color2idx = []
        car_idx = 9
        sign_idx = 7
        fence_idx = 8
        tree_idx=6
        for index, (class_name, color) in enumerate(camvid_color_encoding.items()):
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
        
    def channels(self):
        """
            Returns: int: number of output segmentation channels 
                corresponding to camvid_color_encoding dictionary
        """
        return 13

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, mask) where mask is stack of the masks, corresponding segmentation classes.
        """
        if (self.multiplier):
            index = index % self.maxLen
        img = Image.open(self.X_train[index])
        i_width, i_height = img.size
        if (self.img_resize is not None):
            r_height = self.img_resize[1]
            r_width = self.img_resize[0]
            if (i_width != r_width or i_height != r_height):
                img = transformer.center_cropping_resize(img, self.img_resize)
                i_width, i_height = img.size
        img = np.asarray(img.convert("RGB"), dtype=np.uint8)
        mask = self.readMask(index)
        if (self.img_resize is not None):
            m_height = mask.shape[1]
            m_width = mask.shape[2]
            if (i_width != m_width or i_height != m_height):
                mask = mask.transpose((1, 2, 0))
                mask = cv2.resize(mask, (i_width, i_height))
                mask = mask.transpose((2, 0, 1))

        if self.X_transform:
            mask = mask.transpose((1, 2, 0))
            img, mask = self.X_transform(img, mask)
            mask = mask.transpose((2, 0, 1))

        img = transformer.image_to_tensor(img)
        mask = transformer.mask_to_tensor(mask, self.threshold)
        return img, mask
        
    def __len__(self):
        assert len(self.X_train) == len(self.y_train_masks)
        if (self.multiplier):
            return len(self.X_train) * self.multiplier
        else:
            return len(self.X_train)

    def readMask(self, index):
        fName = self.y_train_masks[index]
        img = Image.open(fName)
        i_width, i_height = img.size
        
        img = np.asarray(img.convert("RGB"), dtype=np.uint8)
        img = np.transpose(img, (2, 0, 1))
        num_classes = self.cannels()
        mask = np.zeros((num_classes, i_height, i_width), dtype=np.float32)
        for e in self.color2idx:
            color = e[0]
            m_idx = np.logical_and(np.logical_and(img[0] == color[0], img[1] == color[1]), img[2] == color[2])
            mask[e[1], m_idx] = 1.
        
        return mask  

class TrainKittiDataset(data.Dataset):
    def __init__(self, X_data, y_data, img_resize,
                 X_transform=None, threshold=0.5, multiplier = None):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()

            Args:
                X_data (list): List of paths to the training images
                y_data (list, optional): List of paths to the target images
                img_resize (tuple): Tuple containing the new size of the images
                X_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                threshold (float): The threshold used to consider the mask present or not
                multiplier (int, optional): increases dataset size by multiplier, 
                    data lists used cyclically
        """
        self.threshold = threshold
        self.X_train = X_data
        self.y_train_masks = y_data
        self.X_transform = X_transform
        self.multiplier = multiplier
        self.maxLen = len(self.X_train)
        self.img_resize = img_resize
        
    def __len__(self):
        assert len(self.X_train) == len(self.y_train_masks)
        if (self.multiplier):
            return len(self.X_train) * self.multiplier
        else:
            return len(self.X_train)
            
    def channels(self):
        """
            Returns: int: number of output segmentation channels 
                corresponding to kitti_idx_encoding dictionary
        """
        return 13
        
    def readMask(self, index):
        fName = self.y_train_masks[index]
        img = cv2.imread(fName, cv2.IMREAD_GRAYSCALE)

        i_width, i_height = img.shape[1], img.shape[0]

        num_classes = self.cannels()
        mask = np.zeros((num_classes, i_height, i_width), dtype=np.float32)
        for i, indexes in kitti_idx_encoding.items():
            for idx in indexes:
                m_idx = (img == idx)
                mask[i, m_idx] = 1.
        
        return mask 
        
    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, mask) where mask is stack of the masks, corresponding segmentation classes.
        """
        if (self.multiplier):
            index = index % self.maxLen
        img = Image.open(self.X_train[index])
        i_width, i_height = img.size
        if (self.img_resize is not None):
            r_height = self.img_resize[1]
            r_width = self.img_resize[0]
            if (i_width != r_width or i_height != r_height):
                img = transformer.center_cropping_resize(img, self.img_resize)
                i_width, i_height = img.size
        img = np.asarray(img.convert("RGB"), dtype=np.uint8)
        mask = self.readMask(index)
        if (self.img_resize is not None):
            m_height = mask.shape[1]
            m_width = mask.shape[2]
            if (i_width != m_width or i_height != m_height):
                mask = mask.transpose((1, 2, 0))
                mask = cv2.resize(mask, (i_width, i_height))
                mask = mask.transpose((2, 0, 1))

        if self.X_transform:
            mask = mask.transpose((1, 2, 0))
            img, mask = self.X_transform(img, mask)
            mask = mask.transpose((2, 0, 1))

        img = transformer.image_to_tensor(img)
        mask = transformer.mask_to_tensor(mask, self.threshold)
        return img, mask
        
class TrainImageDataset(data.Dataset):
    def __init__(self, X_data, y_data=None, img_resize=(128, 128),
                 X_transform=None, y_transform=None, threshold=0.5, multiplier = None):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()

            Args:
                X_data (list): List of paths to the training images
                y_data (list, optional): List of paths to the target images
                img_resize (tuple): Tuple containing the new size of the images
                X_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                y_transform (callable, optional): A function/transform that takes in 2 numpy arrays.
                    Assumes X_data and y_data are not None.
                    (train_img, mask_img) and returns a transformed version with the same signature
                threshold (float): The threshold used to consider the mask present or not
                multiplier (int, optional): increases dataset size by multiplier, 
                    data lists used cyclically
        """
        self.threshold = threshold
        self.X_train = X_data
        self.y_train_masks = y_data
        self.img_resize = img_resize
        self.y_transform = y_transform
        self.X_transform = X_transform
        self.multiplier = multiplier
        self.maxLen = len(self.X_train)

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is class_index of the target class.
        """
        if (self.multiplier):
            index = index % self.maxLen
        img = Image.open(self.X_train[index])
        width, height = img.size
        r_height = self.img_resize[1]
        r_width = self.img_resize[0]
        if (width != r_width or height != r_height):
            img = transformer.center_cropping_resize(img, self.img_resize)
        img = np.asarray(img.convert("RGB"), dtype=np.uint8)

        # Pillow reads gifs
        mask = Image.open(self.y_train_masks[index])
        width, height = mask.size
        r_height = self.img_resize[1]
        r_width = self.img_resize[0]
        if (width != r_width or height != r_height):
            mask = transformer.center_cropping_resize(mask, self.img_resize)
        mask = np.asarray(mask.convert("L"), dtype=np.float32)  # GreyScale

        if self.X_transform:
            img, mask = self.X_transform(img, mask)

        if self.y_transform:
            img, mask = self.y_transform(img, mask)

        img = transformer.image_to_tensor(img)
        mask = transformer.mask_to_tensor(mask, self.threshold)
        return img, mask

    def __len__(self):
        assert len(self.X_train) == len(self.y_train_masks)
        if (self.multiplier):
            return len(self.X_train) * self.multiplier
        else:
            return len(self.X_train)
    
class TestImageDataset(data.Dataset):
    def __init__(self, X_data, img_resize=(128, 128), X_transform=None):
        """
            A dataset loader taking images paths as argument and return
            as them as tensors from getitem()
            Args:
                X_data (list): List of paths to the training images
                img_resize (tuple): Tuple containing the new size of the images
        """
        self.img_resize = img_resize
        self.X_train = X_data
        self.X_transform = X_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path = self.X_train[index]
        img = Image.open(img_path)
        img = transformer.center_cropping_resize(img, self.img_resize)
        img = np.asarray(img.convert("RGB"), dtype=np.uint8)
        
        if self.X_transform:
            img, mask = self.X_transform(img, img)

        img = transformer.image_to_tensor(img)
        return img, img_path.split("/")[-1]

    def __len__(self):
        return len(self.X_train)
        
def checkData():
    print('No op')
	
if __name__== '__main__':
    checkData()
