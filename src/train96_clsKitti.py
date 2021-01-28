from nn.classifier import SegmentationClassifier
import nn.unet_cls as unet
from data.fetcher import TrainDataFetcher
from nn.train_callbacks import ModelSaverCallback
from data.dataset_seg import TrainKittiDataset

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch.optim as optim

import img.augmentation as aug

import os
import time
from multiprocessing import cpu_count

import numpy as np
from PIL import Image


# Hyperparameters
img_resize = (500, 374)
batch_size = 5
epochs = 50
threshold = 0.5
validation_size = 0.1

train_path = '../../../data/kitti/image_2_5'
train_mask_path = '../../../data/kitti/semantic_5'

def modelName():
    return 'unet96cls_kitti'
 
"""
    main entry, starts training of UNet network
"""   
def main():
    np.random.seed(9)
    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Download the datasets
    ds_fetcher = TrainDataFetcher(train_path, train_mask_path, checkImage = False)
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(validation_size=validation_size)

    # Training callbacks
    model_saver_cb = ModelSaverCallback(os.path.join(script_dir, '../output/models/' + modelName()), verbose=True)


    train_ds = TrainKittiDataset(X_train, y_train, img_resize, X_transform=aug.augment_img8, multiplier = 4, threshold=threshold)
    train_loader = DataLoader(train_ds, batch_size,
                              sampler=RandomSampler(train_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)

    valid_ds = TrainKittiDataset(X_valid, y_valid, img_resize, threshold=threshold)
    valid_loader = DataLoader(valid_ds, batch_size,
                              sampler=SequentialSampler(valid_ds),
                              num_workers=threads,
                              pin_memory=use_cuda)
                              
    net = unet.UNet96cls((3, img_resize[0], img_resize[1]), out_channels=train_ds.channels())
    classifier = SegmentationClassifier(net, epochs)
    optimizer = optim.Adadelta(net.parameters())

    print(time.strftime('%D %X'), 'Train model', modelName(), 'batch_size', batch_size, 'out_channels', train_ds.channels(), 'img_resize', img_resize,
          'train_path', train_path, 'train_mask_path', train_mask_path)
    print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    callbacks=[model_saver_cb]
    classifier.train(train_loader, valid_loader, optimizer, epochs, callbacks=callbacks)

"""
    test method, dataset validation
"""
def test():
    ds_fetcher = TrainDataFetcher(train_path, train_mask_path)
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(validation_size=validation_size)
    train_ds = TrainKittiDataset(X_train, y_train, img_resize,  X_transform=aug.augment_img8, threshold = threshold)
    img, mask = train_ds.__getitem__(1)
    print(type( img ),'img size', img.shape, 'mask size', mask.shape)
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    im = Image.fromarray(img.astype('uint8'), 'RGB')
    im.save("img.png")
    mask = mask.numpy()
    mask = mask[5, :, :]
    mask = mask * 255
    m = Image.fromarray(mask.astype('uint8'))
    m.save("mask.png")
        
if __name__ == "__main__":
    main()
    #test()
