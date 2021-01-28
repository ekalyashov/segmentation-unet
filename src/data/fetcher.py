import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

imageExt = ['jpg', 'png', 'gif', 'JPG', 'PNG', 'GIF']

class TestDataFetcher:
    def __init__(self, test_path):
        """
            A class to load file names of test dataset, check 
            and returns set of file paths.
            Args: 
                test_path: path to folder which contains test image files
        """
        self.test_path = test_path
        
    def get_test_files(self, sample_size):
        """
        Args:
            sample_size (float): Value between 0 and 1, if defined, 
            only part of test files returned, if None, full dataset returned
        Returns:
            list : Returns the test dataset in the form of file paths
        """
        self.test_files = sorted(os.listdir(self.test_path))
        test_files = self.test_files

        if sample_size:
            rnd = np.random.choice(self.test_files, int(len(self.test_files) * sample_size))
            test_files = rnd.ravel()
         
        test_files = [x for x in test_files if self.isImage(x)]
        ret = [None] * len(test_files)
        for i, file in enumerate(test_files):
            ret[i] = self.test_path + "/" + file

        return np.array(ret)
        
    def get_image_matrix(self, image_path):
        img = Image.open(image_path)
        return np.asarray(img, dtype=np.uint8)

    def get_image_size(self, image):
        img = Image.open(image)
        return img.size
        
    def isImage(self, fName):
        split =  os.path.splitext(fName)
        ext = split[1][1:].strip()
        return ext in imageExt
        
class TrainDataFetcher:
    def __init__(self, train_path, train_masks_path, checkImage = True):
        """
            A class to load file names of dataset, check, split 
            and returns train and validation sets of file paths.
            Args: 
                train_path: path to folder which contains train image files
                train_masks_path: path to folder which contains train mask files
                checkImage (boolean): if true added check that data file is Image
        """

        self.train_path = train_path
        self.train_masks_path = train_masks_path
        self.train_files = sorted(os.listdir(self.train_path))
        self.train_masks_files = sorted(os.listdir(self.train_masks_path))
        self.checkImage = checkImage
        
    def get_image_matrix(self, image_path):
        img = Image.open(image_path)
        return np.asarray(img, dtype=np.uint8)

    def get_image_size(self, image):
        img = Image.open(image)
        return img.size        
        
    def isImage(self, fName):
        split =  os.path.splitext(fName)
        ext = split[1][1:].strip()
        return ext in imageExt
        
    def get_train_files(self, validation_size=0.2):
        """
        Args:
            validation_size (float): Value between 0 and 1

        Returns:
            list : Returns the dataset in the form:
                [train_data, train_masks_data, valid_data, valid_masks_data]
        """

        train_files = [x for x in self.train_files if not self.checkImage or self.isImage(x)]
        train_masks_files = [x for x in self.train_masks_files if not self.checkImage or self.isImage(x)]
        
        if validation_size:
            X_train, X_test, y_train, y_test = train_test_split(train_files, train_masks_files, test_size=validation_size)
        else:
            X_train = train_files
            X_test = []
            y_train = train_masks_files
            y_test = []
            
        X_train = np.array([self.train_path + "/" + s for s in X_train])
        X_test = np.array([self.train_path + "/" + s for s in X_test])
        y_train = np.array([self.train_masks_path + "/" + s for s in y_train])
        y_test = np.array([self.train_masks_path + "/" + s for s in y_test])

        return X_train, y_train, X_test, y_test    
        
