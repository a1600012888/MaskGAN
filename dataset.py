from __future__ import print_function, division
import os
import torch
import cv2 as cv
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import random
import time
import warnings
from torch.utils.data.sampler import WeightedRandomSampler

warnings.filterwarnings("ignore")

#data
class Cityscapes_dataset(Dataset):

    def __init__(self, json_file, transform = None):
        self.dataset_json = json_file
        self.load()
        self.transform = transform

    def load(self):
        with open(self.dataset_json, 'r') as fp:
            self.meta_data = json.load(fp)
        return self

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self,idx):
        data_info = self.meta_data[idx]
        ori_img = cv.imread(data_info['img_path'])
        ori_img = np.array(ori_img)
        mask = np.zeros((ori_img.shape[0], ori_img.shape[1], 1))

        sy = np.random.randint(0, 255 - 64)     # np.random does not support multi-thread
        sx = np.random.randint(0, 511 - 128)    # set worker_init_fn for dataloader
        ey = sy + 64
        ex = sx + 128

        mask[sy:ey, sx:ex, :] = 1
        hole_img = np.copy(ori_img)
        hole_img[sy:ey, sx:ex, :] = 255
        hole_img = hole_img * 1.0 / 255
        ori_img = ori_img * 1.0 / 255
        if self.transform:
            hole_img = self.transform(hole_img)
            ori_img = self.transform(ori_img)
            mask = self.transform(mask)
        sample = {'hole_img': hole_img, 'ori_img':ori_img, 'mask':mask}
        return sample

def get_dataloaders(json_filepath, batch_size, shuffle=True):
    data_transforms = transforms.Compose([
            transforms.ToTensor()])
    dataset = Cityscapes_dataset(json_filepath, data_transforms)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=8, worker_init_fn=lambda _: np.random.seed())
    return dataloader


def test():
    val_loader = get_dataloaders('/data1/wurundi/cityscapes/data/val.json', 1, False)
    for i, data in enumerate(val_loader):
        if i > 0:
            break
        ori_img = data['ori_img'][0].numpy() * 255
        hole_img = data['hole_img'][0].numpy() * 255
        mask = data['mask'][0].numpy() * 255
        ori_img = ori_img.astype(np.uint8)
        hole_img = hole_img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        ori_img = np.rollaxis(ori_img, 0, 3)
        hole_img = np.rollaxis(hole_img, 0, 3)
        mask = np.rollaxis(mask, 0, 3)
        mask = mask.reshape((mask.shape[0], mask.shape[1]))
        print(mask.shape)
        ori = Image.fromarray(ori_img)
        ori.save('ori_img.png')
        hole = Image.fromarray(hole_img)
        hole.save('hole_img.png')
        ma = Image.fromarray(mask)
        ma.save('mask.png')
        #cv.imwrite('ori_img.png', ori_img)
        #cv.imwrite('hole_img.png', hole_img)



if __name__=='__main__':
    test()