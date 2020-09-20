# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:18:48 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import numpy as np
import math
import torch
from PIL import Image
from matplotlib import pyplot as plt

import constants as C
import utils

Image.MAX_IMAGE_PIXELS = 1000000000

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
        info = {}

        el_data = self.data[index]  # CHW
        p_data = self.data_path[index]

        if self.train:
            el_data = el_data.transpose((1, 2, 0))  # HWC
            el_data = Image.fromarray(el_data)
            info['positive'] = self.data_positive[index]
        if self.transform is not None:
            el_data = self.transform(el_data)
        if self.train:
            el_target = el_data[[-1], :, :]
            el_data = el_data[:-1, :, :]
            if self.unnormalize_target:
                el_target = utils.unnormalize(el_target)
            if self.target_transform is not None:
                el_target = self.target_transform(el_target)
            info['target'] = el_target
        info.update({'data':el_data, 'path': p_data,})
        return info

    def __len__(self):
        return len(self.data)


def da_fetch(d, arr, device=None):
    def to_device(t):
        if isinstance(t, torch.Tensor) and device is not None:
            t = t.to(device)
        return t
    return [to_device(d[i]) for i in arr]

def pic_size_stats(train=True):
    p_data = utils.listdir(C.p_train_p if train else C.p_test, ignore_debug=True)
    size_list = [math.prod(Image.open(p).size) for p in p_data]
    return size_list

class Img_window():
    def __init__(self, img, size, overlap, iy, ix):
        self.size = size
        self.overlap = overlap
        self.iy = iy
        self.ix = ix
        img_size = Img_window._img_size(img)
        self.img_size = img_size
        y_pos = (size[0] - overlap[0]) * iy
        x_pos = (size[1] - overlap[1]) * ix
        self.y_pos = (y_pos, y_pos+size[0] if y_pos+size[0] < img_size[0] else img_size[0])
        self.x_pos = (x_pos, x_pos+size[1] if y_pos+size[1] < img_size[1] else img_size[1])

        img.load()
        self.band = img.im.bands
        self.a = np.array(img, dtype='uint8')[self.y_pos[0]: self.y_pos[1], self.x_pos[0]: self.x_pos[1]]
        self.img = Image.fromarray(self.a)

    def replace_pic(self, img):
        self.img = img
        self.a = np.array(img, dtype='uint8')
        img.load()
        self.band = img.im.bands

    @staticmethod
    def _img_size(img):
        img_size = img.size
        return (img_size[1], img_size[0])

    @staticmethod
    def param_reformat(img_size, size, overlap):
        # URDL
        if isinstance(size, (float, int)):
            size = [size] * 2
        else:
            assert len(size) == 2
        size = [math.floor(img_size[n] * i) if isinstance(i, float) else i for n, i in enumerate(size)]

        if isinstance(overlap, (float, int)):
            overlap = [overlap] * 2
        else:
            assert len(overlap) == 2
        overlap = [math.floor(size[n] * i) if isinstance(i, float) else i for n, i in enumerate(overlap)]

        assert size[0] <= img_size[0] and size[1] <= img_size[1] and overlap[0] * 2 <= size[0] and overlap[1] * 2 <= size[1]
        return size, overlap

    @staticmethod
    def gen_windows(img, size=0.5, overlap=0.1):
        img_size = Img_window._img_size(img)
        size, overlap = Img_window.param_reformat(img_size, size, overlap)
        img_windows = []
        for iy in range(math.ceil((img_size[1]-overlap[1]) / (size[1]-overlap[1]))):
            img_windows_y = []
            for ix in range(math.ceil((img_size[0]-overlap[0]) / (size[0]-overlap[0]))):
                img_windows_y.append(Img_window(img, size, overlap, iy, ix))
            img_windows.append(img_windows_y)
        return img_windows

    @staticmethod
    def merge_windows(windows, method='mean'):
        img_size = windows[0][0].img_size
        band = windows[0][0].band
        size = windows[0][0].size
        overlap = windows[0][0].overlap
        img_size = (img_size[0], img_size[1], band)
        img_a = np.zeros(img_size, dtype='uint16')
        for w_l in windows:
            for w in w_l:
                img_a[w.y_pos[0]: w.y_pos[1], w.x_pos[0]: w.x_pos[1]] += w.a
        dubble_banner_upos = [(size[0] - overlap[0]) * (i + 1) for i in range(len(windows) - 1)]
        dubble_banner_lpos = [(size[1] - overlap[1]) * (i + 1) for i in range(len(w_l) - 1)]
        for upos in dubble_banner_upos:
            img_a[upos: upos + overlap[0], :] = img_a[upos: upos + overlap[0], :] // 2
        for lpos in dubble_banner_lpos:
            img_a[:, lpos: lpos + overlap[1]] = img_a[:, lpos: lpos + overlap[1]] // 2
        img = Image.fromarray(img_a.astype('uint8'))
        return img

def pic_size_plot():
    train_sizes = pic_size_stats(True)
    test_sizes = pic_size_stats(False)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(np.sqrt(test_sizes), label='Test set', alpha=0.8)
    ax.hist(np.sqrt(train_sizes), label='Tarin set', alpha=0.8)
    ax.set_xlabel('sqrt(SIZE)')
    ax.set_ylabel('Number')
    ax.legend(loc='best')

def _main():
    import torchvision.transforms as T

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

    dataiter = iter(trainloader)
    info = dataiter.next()
    data, target, path = da_fetch(info, ['data', 'target', 'path'])

    utils.seg_imshow(data[0], target[0], path[0], alpha=0.5)

if __name__ == "__main__":
    _main()
    # img = Image.open('./input/test/3004.jpg')
    # w = Img_window.gen_windows(img, size=0.3)
    # img2 = Img_window.merge_windows(w)
    # img2.save('./test_3004.jpg')

