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

from load_data import SEED_data
from models.fastfcn import FastFCN as FastFCN
import utils
from constants import device, p_model_param

# In[0]
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

test_set = SEED_data(train=True, transform=transform)

testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1,
    # batch_size=5,
    shuffle=False,
    num_workers=0,
    # drop_last=True,
    # collate_fn=utils.simple_list_collate,
)

# In[1]
model = FastFCN(n_class=1).to(device)
model.load_state_dict(torch.load(p_model_param, map_location=device))

# In[2]
model.eval()
with torch.no_grad():
    for batch, (X_test, _, p_data, _) in tqdm.tqdm(enumerate(testloader)):
        p_data = p_data[0]
        X_test = X_test.to(device)
        X_test_dup = torch.cat((X_test, X_test), dim=0)
        y_test_hat = model(X_test_dup)[0, :, :, :]
        y_pred = (torch.sigmoid(y_test_hat) > 0.5).numpy()
        y_pred_a = y_pred[0, :, :]
        # y_pred_a.dtype = 'uint8'
        img_pred = Image.fromarray((y_pred_a.astype("uint8") * 255), mode="L")
        img_pred.save(utils.gen_p_pred_mask(p_data, "pred"))
        # img_pred.show()
        pass
        # break
# In[1]
# In[1]
