# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:09:29 2020

@author: PENG Feng
@email:  im.pengf@outlook.com
"""

from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None
import numbers
from torchvision.transforms import RandomRotation, RandomResizedCrop, functional as F


class RGBA_RandomRotation(RandomRotation):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        super(RGBA_RandomRotation, self).__init__(
            degrees, resample, expand, center, fill
        )

        if isinstance(fill, numbers.Number):
            self.fill = (fill, fill)
        else:
            if len(fill) != 2:
                raise ValueError("If fill is a sequence, it must be of len 2.")
            self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)
        r, g, b, a = img.split()
        r = F.rotate(r, angle, self.resample, self.expand, self.center, self.fill[0])
        g = F.rotate(g, angle, self.resample, self.expand, self.center, self.fill[0])
        b = F.rotate(b, angle, self.resample, self.expand, self.center, self.fill[0])
        a = F.rotate(a, angle, self.resample, self.expand, self.center, self.fill[1])
        img = Image.merge("RGBA", (r, g, b, a))
        return img


class RGBA_RandomResizedCrop(RandomResizedCrop):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=Image.BILINEAR,
    ):
        super(RGBA_RandomResizedCrop, self).__init__(size, scale, ratio, interpolation)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        r, g, b, a = img.split()
        r = F.resized_crop(r, i, j, h, w, self.size, self.interpolation)
        g = F.resized_crop(g, i, j, h, w, self.size, self.interpolation)
        b = F.resized_crop(b, i, j, h, w, self.size, self.interpolation)
        a = F.resized_crop(a, i, j, h, w, self.size, self.interpolation)
        img = Image.merge("RGBA", (r, g, b, a))
        return img
