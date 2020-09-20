# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:24:51 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""
import re
from matplotlib import pyplot as plt
import pandas as pd

import constants as C

info_dicts = []
re_pattern = r"Epoch (?P<epoch>[\d]+), batch (?P<batch>[\d]+): loss = (?P<loss>[\d\.]+), t_loss = (?P<test_loss>[\d\.]+), t_acc = (?P<test_acc>[\d\.]+) \((?P<time>[\d\.]+)s\).*"
with open(C.p_log, "r") as f:
    for line in f.readlines():
        info = re.match(re_pattern, line)
        info_dicts.append(
            {
                "epoch": int(info.group("epoch")),
                "batch": int(info.group("batch")),
                "loss": float(info.group("loss")),
                "test_loss": float(info.group("test_loss")),
                "test_acc": float(info.group("test_acc")),
            }
        )


df = pd.DataFrame.from_dict(info_dicts)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(df["epoch"], 1 - df["test_loss"])
ax.set_xlabel("Epoch")
ax.set_ylabel("Dice score")
ax.set_title("FastFCN model")
