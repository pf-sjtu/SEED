# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 21:41:15 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import utils

from loss_fn import dice_coeff


def tensor2cv2(t):
    cv_pics = []
    if len(t.size()) < 4:
        t = t.unsqueeze(0)
    for n in range(t.size(0)):
        cv_pic = np.transpose(t[n, :, :, :].numpy(), (1, 2, 0))  # CHW -> HWC
        if cv_pic.dtype != "uint8":
            cv_pic = (cv_pic * 255).astype("uint8")
        if t.size(1) == 3:
            cv_pic = cv_pic[:, :, [2, 1, 0]]  # RGB -> BGR
        # else:
        #     cv_pic = np.concatenate((cv_pic, cv_pic, cv_pic), axis=2)
        cv_pics.append(cv_pic)
    return cv_pics


def cv22tensor(cv2_pics):
    tensor = None
    for cv2_pic in cv2_pics:
        if len(cv2_pic.shape) == 2:
            cv2_pic = cv2_pic[:, :, None]
        cv2_pic = np.transpose(cv2_pic, (2, 0, 1))
        t = torch.tensor(cv2_pic / 255).unsqueeze(0)
        if t.size(2) == 3:
            t = t[:, [2, 1, 0], :, :]
        if tensor is not None:
            tensor = torch.cat((tensor, t), 0)
        else:
            tensor = t
    return tensor

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
        ]
    )

    for n in range(2000, 2007):
        p_masks = [
            "./input/positive/{}_mask.jpg".format(n),
            "./pred50_512/{}_mask.jpg".format(n),
        ]

        masks = [utils.binary(transform(Image.open(i)).unsqueeze(0)) for i in p_masks]
        dice = dice_coeff(masks[0], masks[1], sigmoid=False)
        print(
            "pic: {}, iter: -1, dice: {:.4f}, dice_loss: {:.4f}.".format(n, dice, 1 - dice)
        )

        mask2 = tensor2cv2(masks[1])[0]

        kernel = np.ones((5, 5), np.uint8)
        mask2_erosion1 = cv2.erode(mask2, kernel, iterations=1)
        mask2_erosion10 = cv2.dilate(mask2_erosion1, kernel, iterations=1)
        for i in range(3):
            mask2_erosion10 = cv2.erode(
                mask2_erosion10, kernel, iterations=1
            )
            mask2_erosion10 = cv2.dilate(mask2_erosion10, kernel, iterations=1)
            dice = dice_coeff(masks[0], cv22tensor([mask2_erosion10]), sigmoid=False)
            print(
                "pic: {}, iter: {}, dice: {:.4f}, dice_loss: {:.4f}.".format(
                    n, i, dice, 1 - dice
                )
            )
            # cv2.imshow("mask2_" + str(i), mask2_erosion10)

        # cv2.imshow("mask2", mask2)
        # # cv2.imshow("mask2_erosion1", mask2_erosion1)
        # cv2.imshow("mask2_erosion10", mask2_erosion10)
        # cv2.waitKey()
        # cv2.destroyAllWindows() # important part!
