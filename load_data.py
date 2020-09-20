# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:18:48 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

# import pandas as pd
import numpy as np
import math

# import matplotlib.pyplot as plt
import torch

# import torchvision
import torchvision.transforms as T
from PIL import Image

import constants as C
import utils
from models.fcn8s import FCN8s as fcn
from loss_fn import DiceLoss

# In[0]
class SEED_data(torch.utils.data.Dataset):
    def __init__(
        self, train=True, transform=None, target_transform=None, unnormalize_target=True, part=(1,1), max_len=-1, overlap_pct=0.1
    ):
        self.train = train  # training set or test set
        self.part = part
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.data_positive = None
        self.data = []
        self.data_path = []
        self.unnormalize_target = unnormalize_target

        if train:
            p_data_p_list = utils.listdir(C.p_train_p)
            p_data_p_list = self._cut_list(p_data_p_list, self.part)
            if C.NO_NEGATIVE:
                p_data_n_list = []
            else:
                p_data_n_list = utils.listdir(C.p_train_n)
            # p_target_p_list = [utils.gen_p_mask(i) for i in p_data_list]
            self.data_positive = []
            for p_data_p in p_data_p_list:
                p_target_p = utils.gen_p_mask(p_data_p)
                el_data = np.array(Image.open(p_data_p), dtype=np.uint8)
                el_target = np.array(Image.open(p_target_p), dtype=np.uint8)
                data_target = np.concatenate((el_data, el_target[:, :, None]), axis=2)
                data_target = np.transpose(data_target, (2, 0, 1))  # CHW
                self.data.append(data_target)
                self.data_positive.append(True)
            for p_data_n in p_data_n_list:
                el_data = np.array(Image.open(p_data_n), dtype=np.uint8)
                el_target = np.zeros(el_data.shape[:2])
                data_target = np.concatenate((el_data, el_target[:, :, None]), axis=2)
                data_target = np.transpose(data_target, (2, 0, 1))
                self.data.append(data_target)
                self.data_positive.append(False)
            self.data_path = p_data_p_list + p_data_n_list
        else:
            p_data_list = utils.listdir(C.p_test)
            p_data_list = self._cut_list(p_data_list, self.part)
            self.data = [
                np.array(Image.open(p_data), dtype=np.uint8) for p_data in p_data_list
            ]
            self.data_path = p_data_list

    def _cut_list(self, l, part=(1,1)):
        return l[math.floor(len(l)*(part[0]-1)/part[1]): math.floor(len(l)*(part[0])/part[1])]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        el_data = self.data[index]  # CHW
        p_data = self.data_path[index]

        if self.train:
            el_data = el_data.transpose((1, 2, 0))  # HWC
            el_data = Image.fromarray(el_data)
            positive = self.data_positive[index]
        else:
            positive = True
        if self.transform is not None:
            el_data = self.transform(el_data)
        if self.train:
            el_target = el_data[[-1], :, :]
            el_data = el_data[:-1, :, :]
            if self.unnormalize_target:
                el_target = utils.unnormalize(el_target)
            if self.target_transform is not None:
                el_target = self.target_transform(el_target)
        else:
            el_target = 0
        return el_data, el_target, p_data, positive

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5), (0.5))])

    # 5GB memery
    trainset = SEED_data(train=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        # batch_size=5,
        shuffle=True,
        num_workers=0,
        # collate_fn=utils.simple_list_collate,
    )

    valloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1,
        # batch_size=5,
        shuffle=True,
        num_workers=0,
        # collate_fn=utils.simple_list_collate,
    )

    # testset = Numpy_data(train=False, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=5,
    #                                          shuffle=False, num_workers=0)

    dataiter = iter(trainloader)

    data, target, p_data, positive = dataiter.next()

    # In[1]
    i = 0
    utils.seg_imshow(data[i], target[i], p_data[i], alpha=0.5)

    # In[2]
