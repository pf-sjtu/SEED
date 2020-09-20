# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:12:19 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import tqdm
from PIL import Image
import math

from load_data import SEED_data, da_fetch, Img_window
from models.fastfcn import FastFCN as FastFCN
import utils
from constants import device, p_model_param

# In[0]
train = False
size_thd = 512
pred_batch = 4

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

test_set = SEED_data(train=train, transform=transform if train else None)

testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1,
    # batch_size=5,
    shuffle=False,
    num_workers=0,
    # drop_last=True,
    collate_fn=utils.simple_list_collate,
)

# In[1]
model = FastFCN(n_class=1).to(device)
model.load_state_dict(torch.load(p_model_param, map_location=device))

# In[2]

model.eval()
with torch.no_grad():
    for batch, info_test in tqdm.tqdm(enumerate(testloader)):
        [X_test], [p_data] = da_fetch(info_test, ["data", "path"])
        if X_test.size[0] > size_thd and X_test.size[1] > size_thd:
            windows = Img_window.gen_windows(X_test, size=size_thd, overlap=0.1)
            X_test_l = []
            for w_l in windows:
                for w in w_l:
                    X_test_l.append(transform(w.img).unsqueeze(0))
            # img2 = Img_window.merge_windows(X_test)
        else:
            X_test_l = [transform(X_test).unsqueeze(0)]
        if len(X_test_l) % pred_batch == 1:
            X_test_l.append(X_test_l[-1])
        y_pred = []
        for i in range(math.ceil(len(X_test_l) / pred_batch)):
            X_test_batch = X_test_l[
                pred_batch * i : pred_batch * (i + 1)
                if pred_batch * (i + 1) < len(X_test_l)
                else None
            ]
            X_test = torch.cat(X_test_batch, dim=0).to(device)
            y_test_hat = model(X_test)
            y_pred.append(y_test_hat)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred = (torch.sigmoid(y_pred).numpy() * 255).astype("uint8")
        if y_pred.shape[0] > 2:
            counter = 0
            for w_l in windows:
                for w in w_l:
                    img_part = Image.fromarray(y_pred[counter, 0, :, :], mode="L")
                    w.replace_pic(img_part)
                    counter += 1
            img_pred = Img_window.merge_windows(windows, "max")
            y_pred_a = np.array(img_pred, dtype="uint8")
        else:
            y_pred_a = y_pred[0, 0, :, :]
        y_pred_a = np.where(y_pred_a > 255 // 2, 255, 0).astype("uint8")
        img_pred = Image.fromarray(y_pred_a, mode="L")
        # y_pred_a = y_pred[0, :, :]
        # # y_pred_a.dtype = 'uint8'
        # img_pred = Image.fromarray((y_pred_a), mode="L")
        # # img_pred.save(utils.gen_p_pred_mask(p_data, "pred"))
        img_pred.show()
        # # windows[0][0].img.show()
        pass
        break
# In[1]
# In[1]
