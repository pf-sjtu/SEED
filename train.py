# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:18:48 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import torch
import numpy as np

# import torchvision
import torchvision.transforms as T
import my_trans as T_m

import constants as C
from constants import device
import utils
from models.fastfcn import FastFCN as FastFCN
from loss_fn import DiceLoss
from load_data import SEED_data, da_fetch
import tqdm

transform = T.Compose(
    [
        # T.RandomRotation((40,50), fill=255),
        T_m.RGBA_RandomRotation(90, fill=(255, 255)),
        # T.RandomCrop(
        #     400, pad_if_needed=True, fill=255, padding_mode="constant"
        # ),
        T_m.RGBA_RandomResizedCrop(300, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize((0.5), (0.5)),
    ]
)

# 5GB memery
whole_set = SEED_data(train=True, transform=transform)
train_size = int(0.9 * len(whole_set))
validate_size = len(whole_set) - train_size
train_set, validate_set = torch.utils.data.random_split(
    whole_set, [train_size, validate_size]
)

trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=3,
    # batch_size=5,
    shuffle=True,
    num_workers=0,
    # drop_last=True,
    # collate_fn=utils.simple_list_collate,
)

valloader = torch.utils.data.DataLoader(
    validate_set,
    batch_size=3,
    # batch_size=5,
    shuffle=True,
    num_workers=0,
    # collate_fn=utils.simple_list_collate,
)

# In[2]
dataiter = iter(trainloader)

data, target, p_data, positive = dataiter.next()

# In[1]
i = 0
utils.seg_imshow(data[i], target[i], p_data[i], alpha=0.5)


# In[2]
# model = models.SegNet(3, 2)
# model = FCN(n_class=1).to(device)
model = FastFCN(n_class=1).to(device)

# In[2]
import time
import torch.optim as optim

loss_func = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.RMSprop(
#     model.parameters(), lr=0.001, alpha=0.9, eps=1e-8, weight_decay=0.0
# )
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.5,
#     patience=3,
#     verbose=False,
#     threshold=1e-4,
#     threshold_mode="rel",
#     cooldown=0,
#     min_lr=0.00001,
#     eps=1e-8,
# )

# In[2]
epoch = 64
verbose = 1
log = ""

saved_flag = False
t1 = time.perf_counter()
best_test_loss = 999.0
for e in range(epoch):
    running_loss = 0.0
    # running_loss_init = False
    for batch, info_train in tqdm.tqdm(enumerate(trainloader)):
        with torch.no_grad():
            X_train, y_train = da_fetch(info_train, ["data", "target"], device=device)
            utils.binary(y_train)
        optimizer.zero_grad()
        y_train_hat = model(X_train)
        loss = loss_func(y_train_hat, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pass
    model.eval()
    with torch.no_grad():
        test_acc, test_loss = 0.0, 0.0
        for batch_test, info_test in enumerate(valloader):
            X_test, y_test = da_fetch(info_test, ["data", "target"], device=device)
            utils.binary(y_test)
            y_test_hat = model(X_test)
            test_acc += torch.mean(
                ((torch.sigmoid(y_test_hat) > 0.5) == (y_test > 0.5)).float()
            )
            test_loss += loss_func(y_test_hat, y_test)
        test_acc /= len(valloader)
        test_loss /= len(valloader)
        running_loss /= len(trainloader)
        if test_loss < best_test_loss:
            torch.save(model.state_dict(), C.p_model_param)
            best_test_loss = test_loss
            saved_flag = True
        else:
            saved_flag = False
        info = "Epoch {}, batch {}: loss = {:.4f}, t_loss = {:.4f}, t_acc = {:.4f} ({:.1f}s){}".format(
            e,
            batch,
            running_loss,
            test_loss,
            test_acc,
            time.perf_counter() - t1,
            " saved" if saved_flag else "",
        )
        log += info + "\n"
        print(info)
        with open(C.p_log, "w") as f:
            f.write(log)
        running_loss = 0.0
    model.train()
