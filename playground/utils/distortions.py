import ctypes
import io
import math
import random

import kornia
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F

from utils.utils_distortions import curves, filter2D, fspecial


def gaussian_blur(x: torch.Tensor, blur_sigma: int = 0.1) -> torch.Tensor:
    if blur_sigma == 0:
        return x
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def lens_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return x
    h = fspecial('disk', radius)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def motion_blur(x: torch.Tensor, radius: int, angle: bool = None) -> torch.Tensor:
    if radius == 0:
        return x
    if angle is None:
        angle = random.randint(0, 180)
    h = fspecial('motion', radius, angle)
    h = torch.from_numpy(h.copy()).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def color_saturation1(x: torch.Tensor, factor: int) -> torch.Tensor:
    factor = 1 - factor
    if factor == 1:
        return x
    x = x[[2, 1, 0], ...]
    hsv = kornia.color.rgb_to_hsv(x)
    hsv[1, ...] *= factor
    y = kornia.color.hsv_to_rgb(hsv)
    return y[[2, 1, 0], ...]


def color_saturation2(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 0:
        return x
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    lab[1:3, ...] = lab[1:3, ...] * factor
    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255) / 255.
    return y[[2, 1, 0], ...]


def brighten(x: torch.Tensor, amount: float) -> torch.Tensor:
    if amount == 0.0:
        return x
    x = x[[2, 1, 0]]
    lab = kornia.color.rgb_to_lab(x)

    l = lab[0, ...] / 100.
    l_ = curves(l, 0.5 + amount / 2)
    lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 + amount / 2)

    j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)

    y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def darken(x: torch.Tensor, amount: float, dolab: bool = False) -> torch.Tensor:
    if amount == 0.0:
        return x
    x = x[[2, 1, 0], :, :]
    lab = kornia.color.rgb_to_lab(x)
    if dolab:
        l = lab[0, ...] / 100.
        l_ = curves(l, 0.5 + amount / 2)
        lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 - amount / 2)

    if dolab:
        j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)
        y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def perspective_top(x: torch.Tensor, amount: float = 0.6) -> torch.Tensor:
    if amount == 0.0:
        return x
    w, h = x.shape[-1], x.shape[1]
    src_points = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]], dtype=torch.float32)
    dst_points = src_points.clone().detach()
    dst_points[2][0] -= w * amount  # Move top left point to the left
    dst_points[3][0] += w * amount  # Move top right point to the right
    y = TF.perspective(x, src_points, dst_points)
    return y

def perspective_bottom(x: torch.Tensor, amount: float = 0.6) -> torch.Tensor:
    if amount == 0.0:
        return x
    w, h = x.shape[-1], x.shape[1]
    src_points = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]], dtype=torch.float32)
    dst_points = src_points.clone().detach()
    dst_points[0][0] -= w * amount  # Move bottom left point to the left
    dst_points[1][0] += w * amount  # Move bottom right point to the right
    y = TF.perspective(x, src_points, dst_points)
    return y

def perspective_left(x: torch.Tensor, amount: float = 0.6) -> torch.Tensor:
    if amount == 0.0:
        return x
    w, h = x.shape[-1], x.shape[1]
    src_points = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]], dtype=torch.float32)
    dst_points = src_points.clone().detach()
    dst_points[1][1] -= h * amount  # Move top right point upwards
    dst_points[3][1] += h * amount  # Move bottom right point downwards
    y = TF.perspective(x, src_points, dst_points)
    return y

def perspective_right(x: torch.Tensor, amount: float = 0.6) -> torch.Tensor:
    if amount == 0.0:
        return x
    w, h = x.shape[-1], x.shape[1]
    src_points = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]], dtype=torch.float32)
    dst_points = src_points.clone().detach()
    dst_points[0][1] -= h * amount  # Move top left point upwards
    dst_points[2][1] += h * amount  # Move bottom left point downwards
    y = TF.perspective(x, src_points, dst_points)
    return y