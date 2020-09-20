# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:19:46 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

import os
import torch

DEBUG = True
DEBUG_PIC_LIMIT = 10
NO_NEGATIVE = True

p_wd = "."
os.chdir(p_wd)

p_train_p = "./input/positive"
p_train_n = "./input/negative"
p_test = "./input/test"
p_model_param = "./model50_512_dict.torch"
p_log = "./train.log"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
