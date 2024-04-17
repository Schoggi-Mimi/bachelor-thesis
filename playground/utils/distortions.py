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
from torchvision import transforms
from torchvision.io.image import decode_jpeg, encode_jpeg

from utils.utils_distortions import curves, filter2D, fspecial


def gaussian_blur(x: torch.Tensor, blur_sigma: int = 0.1) -> torch.Tensor:
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def lens_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    h = fspecial('disk', radius)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def motion_blur(x: torch.Tensor, radius: int, angle: bool = None) -> torch.Tensor:
    if angle is None:
        angle = random.randint(0, 180)
    h = fspecial('motion', radius, angle)
    h = torch.from_numpy(h.copy()).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def color_diffusion(x: torch.Tensor, amount: int) -> torch.Tensor:
    blur_sigma = 1.5 * amount + 2
    scaling = amount
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)

    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    diff_ab = filter2D(lab[:, 1:3, ...], h.unsqueeze(0))
    lab[:, 1:3, ...] = diff_ab * scaling

    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255.) / 255.
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def color_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    def perc(x, perc):
        xs = torch.sort(x)
        i = len(xs) * perc / 100.
        i = max(min(i, len(xs)), 1)
        v = xs[round(i - 1)]
        return v

    gray = kornia.color.rgb_to_grayscale(x)
    gradxy = kornia.filters.spatial_gradient(gray.unsqueeze(0), 'diff')
    e = torch.sum(gradxy ** 2, 2) ** 0.5

    fs = 2 * math.ceil(2 * 4) + 1
    h = fspecial('gaussian', (fs, fs), 4)
    h = torch.from_numpy(h).float()

    e = filter2D(e, h.unsqueeze(0))

    mine = torch.min(e)
    maxe = torch.max(e)

    if mine < maxe:
        e = (e - mine) / (maxe - mine)

    percdev = [1, 1]
    valuehi = perc(e, 100 - percdev[1])
    valuelo = 1 - perc(1 - e, 100 - percdev[0])

    e = torch.max(torch.min(e, valuehi), valuelo)

    channel = 1
    g = x[channel, :, :]
    a = np.random.random((1, 2))
    amount_shift = np.round(a / (np.sum(a ** 2) ** 0.5) * amount)[0].astype(int)

    y = F.pad(g, (amount_shift[0], amount_shift[0]), mode='replicate')
    y = F.pad(y.transpose(1, 0), (amount_shift[1], amount_shift[1]), mode='replicate').transpose(1, 0)
    y = torch.roll(y, (amount_shift[0], amount_shift[1]), dims=(0, 1))

    if amount_shift[1] != 0:
        y = y[amount_shift[1]:-amount_shift[1], ...]
    if amount_shift[0] != 0:
        y = y[..., amount_shift[0]:-amount_shift[0]]

    yblend = y * e + x[channel, ...] * (1 - e)
    x[channel, ...] = yblend

    return x


def color_saturation1(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    hsv = kornia.color.rgb_to_hsv(x)
    hsv[1, ...] *= factor
    y = kornia.color.hsv_to_rgb(hsv)
    return y[[2, 1, 0], ...]


def color_saturation2(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    lab[1:3, ...] = lab[1:3, ...] * factor
    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255) / 255.
    return y[[2, 1, 0], ...]


def jpeg2000(x: torch.Tensor, ratio: int) -> torch.Tensor:
    ratio = int(ratio)
    compression_params = {
        'quality_mode': 'rates',
        'quality_layers': [ratio],  # Compression ratio
        'num_resolutions': 8,  # Number of wavelet decompositions
        'prog_order': 'LRCP',  # Progression order: Layer-Resolution-Component-Position
    }

    # Compress the image and save it using the JPEG2000 format
    x *= 255.
    x = x.byte().cpu().numpy()

    x = Image.fromarray(x.transpose(1, 2, 0), 'RGB')

    with io.BytesIO() as output:
        x.save(output, format='JPEG2000', **compression_params)
        compressed_data = output.getvalue()

    y = Image.open(io.BytesIO(compressed_data))
    y = transforms.ToTensor()(y)

    return y


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    x *= 255.
    y = encode_jpeg(x.byte().cpu(), quality=quality)
    y = (decode_jpeg(y) / 255.).to(torch.float32)
    return y


def brighten(x: torch.Tensor, amount: float) -> torch.Tensor:
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


def mean_shift(x: torch.Tensor, amount: float) -> torch.Tensor:
    x = x[[2, 1, 0], :, :]

    y = torch.clamp(x + amount, 0, 1)
    return y[[2, 1, 0]]


def color_block(x: torch.Tensor, pnum: int) -> torch.Tensor:
    patch_size = [32, 32]

    c, w, h = x.shape

    y = x

    h_max = h - patch_size[0]
    w_max = w - patch_size[1]

    for i in range(pnum):
        color = np.random.random(3)
        px = math.floor(random.random() * w_max)
        py = math.floor(random.random() * h_max)
        patch = torch.ones((3, patch_size[0], patch_size[1]))
        for j in range(3):
            patch[j, ...] *= color[j]
        y[:, px:px + patch_size[0], py:py + patch_size[1]] = patch

    return y

def perspective(x: torch.Tensor, direction: int) -> torch.Tensor:
    amount = 0.6
    w, h = x.shape[-1], x.shape[1]
    src_points = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]], dtype=torch.float32)
    dst_points = src_points.clone().detach()
    if direction == 0: # Top
        dst_points[2][0] -= w * amount  # Move top left point to the left
        dst_points[3][0] += w * amount  # Move top right point to the right
    elif direction == 1: # Bottom
        dst_points[0][0] -= w * amount  # Move bottom left point to the left
        dst_points[1][0] += w * amount  # Move bottom right point to the right
    elif direction == 2: # Left
        dst_points[1][1] -= h * amount  # Move top right point upwards
        dst_points[3][1] += h * amount  # Move bottom right point downwards
    elif direction == 3: # Right
        dst_points[0][1] -= h * amount  # Move top left point upwards
        dst_points[2][1] += h * amount  # Move bottom left point downwards
    y = TF.perspective(x, src_points, dst_points)
    return y