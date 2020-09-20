# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:04:57 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""
import torch
import torch.nn as nn

# Dice系数
def dice_coeff(predict, target, epsilon=1e-5, sigmoid=True):
    target = target.to(predict.device)
    num = predict.size(0)

    if sigmoid:
        pre = torch.sigmoid(predict).view(num, -1)
    else:
        pre = predict.view(num, -1)
    tar = target.view(num, -1)

    intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
    union = pre.sum() + tar.sum()

    coeff = 2 * (intersection + epsilon) / (union + epsilon)

    return coeff


# Dice损失函数
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predict, target):
        assert (
            predict.size() == target.size()
        ), "the size of predict and target must be equal."
        loss = 1 - dice_coeff(predict, target, self.epsilon)
        return loss


if __name__ == "__main__":
    loss = DiceLoss()
    predict = torch.randn(3, 4, 4)
    target = torch.randn(3, 4, 4)

    score = loss(predict, target)
    print(score)
