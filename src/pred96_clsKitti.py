from nn.classifier import SegmentationClassifier
from nn.test_callbacks import PredictionsKittiSaverCallback, DiceCoefCamVidCallback
import nn.unet_cls as unet

from data.fetcher import TestDataFetcher
from data.dataset_seg import TestImageDataset

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

import time

from multiprocessing import cpu_count

# Hyperparameters
out_channels = 13
img_resize = (500, 374)
batch_size = 2
epochs = 50
threshold = 0.5
validation_size = 0.2

def modelName():
    return 'unet96cls_kitti'

"""
    main entry, performs inference for specified test dataset
"""
def main():
    test_mask_path = None
    
    test_path = '../../../data/kitti/test/image_2'
    out_path = '../output/pred_kitti'

    sample_size = None  # Put None to work on full dataset
    
    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    
    ds_fetcher = TestDataFetcher(test_path)
    full_x_test = ds_fetcher.get_test_files(sample_size)
    origin_img_size = img_resize

    print(time.strftime('%D %X'), 'Prediction, model', modelName(), 'batch_size', batch_size,
          'test_path', test_path, 'test_mask_path', test_mask_path, 'out_path', out_path)
    print('origin_img_size', origin_img_size, 'img_resize', img_resize, 'out_channels', out_channels)

                                                                    
    pmask_saver_cb = PredictionsKittiSaverCallback(test_path, out_path, maskOnly=False)

    net = unet.UNet96cls((3, img_resize[0], img_resize[1]), out_channels=out_channels)
    net.load_state_dict(torch.load('../output/models/' + modelName()))
    net.eval()
    
    classifier = SegmentationClassifier(net, epochs)

    test_ds = TestImageDataset(full_x_test, img_resize, X_transform=None)
    
    test_loader = DataLoader(test_ds, batch_size,
                             sampler=SequentialSampler(test_ds),
                             num_workers=threads,
                             pin_memory=use_cuda)

    if use_cuda:
        net.cuda()
    # Predict & save
    if (test_mask_path is None):
        callbacks=[pmask_saver_cb]
    else:
        dcCallback = DiceCoefCamVidCallback(test_mask_path, origin_img_size, threshold)
        callbacks=[pmask_saver_cb, dcCallback]
    
    classifier.predict(test_loader, callbacks=callbacks)
    if (test_mask_path is not None):
        dcCallback.print_res()
    
if __name__ == "__main__":
    # Workaround for a deadlock issue on Pytorch 0.2.0: https://github.com/pytorch/pytorch/issues/1838
    #multiprocessing.set_start_method('spawn', force=True)
    main()
