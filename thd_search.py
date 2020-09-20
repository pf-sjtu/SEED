# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:12:19 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import torch
import numpy as np
import tqdm
from PIL import Image
import torchvision.transforms as T
import my_trans as T_m

from load_data import SEED_data
from models.fastfcn import FastFCN as FastFCN
import utils
from constants import device, p_model_param

tarin_compare = True
# In[0]
transform = T.Compose(
    [
        T_m.RGBA_RandomRotation(90, fill=(255, 255)),
        T_m.RGBA_RandomResizedCrop(300, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize((0.5), (0.5)),
    ]
)

test_set = SEED_data(train=tarin_compare, transform=transform)

testloader = torch.utils.data.DataLoader(
    test_set, batch_size=1, shuffle=False, num_workers=0,
)

# In[1]
model = FastFCN(n_class=1).to(device)
model.load_state_dict(torch.load(p_model_param, map_location=device))

# In[2]
from loss_fn import dice_coeff
model.eval()
with torch.no_grad():
    for batch, (X_test, y_test, p_data, _) in tqdm.tqdm(enumerate(testloader)):
        p_data = p_data[0]
        X_test = X_test.to(device)
        X_test_dup = torch.cat((X_test, X_test), dim=0)
        y_test_hat = model(X_test_dup)[0, :, :, :]
        y_pred = (torch.sigmoid(y_test_hat) > 0.5)
        y_pred_a = y_pred.numpy().astype("float")[0, :, :]
        y_test_a = y_test.numpy()[0, 0, :, :]
        # y_pred_a.dtype = 'uint8'
        img_pred = Image.fromarray((y_pred_a * 255), mode="L")
        # img_pred.save(utils.gen_p_pred_mask(p_data, "pred"))
        # img_pred.show()
        for thd in np.linspace(0.1, 0.9, 9):
            y_pred = (torch.sigmoid(y_test_hat) > thd)
            dice = dice_coeff(y_test, y_pred, sigmoid=False)
            print('thd: {:.2f}, dice: {:.4f}, dice_loss: {:.4f}.'.format(thd, dice, 1-dice))
        pass
        # break
# In[1]
# In[1]


    # y_pred_a = y_pred[0, :, :]
    # y_pred_a.dtype = 'uint8'
    # img_pred = Image.fromarray((y_pred_a.astype("uint8") * 255), mode="L")
# img_pred.show()