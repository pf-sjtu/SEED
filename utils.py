# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:30:34 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""
import os
import re
from matplotlib import pyplot as plt
import numpy as np
import torch
import hashlib
import gdown

import constants as C


def listdir(p_dir, join=True, exclude_mask=True, ignore_debug=False):
    l = os.listdir(p_dir)
    if exclude_mask:
        l = [i for i in l if ("_mask" not in i) and (" " not in i)]
    l.sort()
    if join:
        l = [os.path.join(p_dir, i) for i in l]
    if not ignore_debug and C.DEBUG:
        l = l[: C.DEBUG_PIC_LIMIT]
    return l


def gen_p_mask(p_img):
    p = r"(?P<p>.*[\\/])(?P<n>\d+)(?P<t>\..+)"
    return re.sub(
        p, lambda x: x.group("p") + x.group("n") + "_mask" + x.group("t"), p_img
    )


def gen_p_pred_mask(p_img, dir="pred"):
    dir = "./" + dir
    if not os.path.isdir(dir):
        os.mkdir(dir)
    filename = re.findall(r"[^/\\]*", p_img)[-2]
    return gen_p_mask("{}/{}".format(dir, filename))


def normalize(img):
    return img * 2 - 1.0


def unnormalize(img):
    return img / 2 + 0.5


def imshow(img, one_channel=False, unnorm=True):
    if one_channel:
        img = img[0, :, :]
        # img = img.mean(dim=0)
    if unnorm:
        img = unnormalize(img)  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def seg_imshow(data, target, p_data, alpha=0):
    target_alpha = normalize(target)
    # target_alpha = torch.zeros_like(target).copy_(target)
    target_alpha[target_alpha < 0] += 2 * alpha
    img = torch.cat([data, target_alpha], dim=0)
    imshow(img, unnorm=True, one_channel=False)
    print(p_data)


def simple_list_collate(batch):
    item_num = len(batch[0])
    l = [[item[i] for item in batch] for i in range(item_num)]
    # data = [item[0] for item in batch]
    # target = [item[1] for item in batch]
    # target = torch.LongTensor(target)
    return l


def md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()


def cached_download(url, path, md5=None, quiet=False, postprocess=None):
    def check_md5(path, md5):
        print("[{:s}] Checking md5 ({:s})".format(path, md5))
        return md5sum(path) == md5

    if os.path.exists(path) and not md5:
        print("[{:s}] File exists ({:s})".format(path, md5sum(path)))
    elif os.path.exists(path) and md5 and check_md5(path, md5):
        pass
    else:
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        gdown.download(url, path, quiet=quiet)

    if postprocess is not None:
        postprocess(path)

    return path


def binary(tensor):
    max_val, min_val = tensor.max(), tensor.min()
    mid = (max_val + min_val) * 0.5
    tensor[tensor >= mid] = max_val
    tensor[tensor < mid] = min_val
    return tensor

# p0 = 3
# k = 10
# e = np.exp(1)
# x = torch.Tensor(np.linspace(0, 100, 100))
# y = k*p0*e**x/(k+p0*(e**x-1))
# plt.plot(x.numpy(), y.numpy())