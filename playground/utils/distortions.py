import math
import random

import kornia
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.nn import functional as F

from utils.utils_distortions import (curves, filter2D, fspecial,
                                     skin_segmentation)


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

def crop_image(image: torch.Tensor, level: int) -> torch.Tensor:
    _, height, width = image.shape
    if level == 0:
        return image  # No cropping for level 0.
    
    crop_ratio = level / 5
    new_height = int(height * (1 - crop_ratio / 2))
    new_width = int(width * (1 - crop_ratio / 2))

    # Crop the image towards the bottom-right corner
    cropped_image = image[:, :new_height, :new_width]
    return cropped_image

def crop_image1(image: torch.Tensor, level: int) -> torch.Tensor:
    _, height, width = image.shape
    if level == 0:
        return image  # No cropping for level 0.

    crop_ratio = level / 5
    new_height = int(height * (1 - crop_ratio / 2))
    new_width = int(width * (1 - crop_ratio / 2))

    # Crop the image towards the bottom-right corner
    cropped_image = image[:, :new_height, :new_width]

    # Resize cropped image back to the original dimensions if necessary
    if new_height != height or new_width != width:
        cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)

    return cropped_image
    
def change_resolution(x: torch.Tensor, distortion_level: float) -> torch.Tensor:
    scale_level = 1 - distortion_level
    if scale_level == 1:
        return x
    _, h, w = x.shape
    new_h, new_w = max(1, int(h * scale_level)), max(1, int(w * scale_level))

    if new_h < 1 or new_w < 1:
        raise ValueError("Scale level too small, resulting dimensions are less than 1 pixel.")

    downsampled = torch.nn.functional.interpolate(x.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
    restored = torch.nn.functional.interpolate(downsampled, size=(h, w), mode='bilinear', align_corners=False)

    return restored.squeeze(0)

def color_block(x: torch.Tensor, amount: float) -> torch.Tensor:
    if amount == 0:
        return x
    skin_mask = skin_segmentation(x)
    background_mean = 1 - skin_mask.mean().item()  # Estimate of background area
    if background_mean < 0.05:  # Only apply distortion if significant background is present
        return x
    x_new = x * ~skin_mask
    amount = amount * background_mean
    _, h, w = x_new.shape
    exclusion_ratio = 0.2  # Exclude central region from distortion
    patch_size = [int(max(32, min(h / 10 * amount, h))), int(max(32, min(w / 10 * amount, w)))]
    
    y = x_new.clone()

    # Convert numpy mask to torch tensor
    mask_tensor = torch.from_numpy(~skin_mask).float()

    h_max = h - patch_size[0]
    w_max = w - patch_size[1]

    # Define central exclusion zone
    central_height = int(h * exclusion_ratio)
    central_width = int(w * exclusion_ratio)
    central_top = (h - central_height) // 2
    central_bottom = central_top + central_height
    central_left = (w - central_width) // 2
    central_right = central_left + central_width

    num_patches = max(1, int(amount * 10))
    attempts = 0
    for _ in range(num_patches):
        while True:
            if h_max <= 0 or w_max <= 0 or attempts > 100:  # Avoid infinite loop by limiting attempts
                break
            px = random.randint(0, w_max)
            py = random.randint(0, h_max)
            # Check if the selected area is in the background and outside the central exclusion zone
            if not (central_left < px < central_right and central_top < py < central_bottom):
                patch_area = mask_tensor[py:py + patch_size[0], px:px + patch_size[1]]
                if torch.all(patch_area == 1):  # Ensure the entire patch is within the background
                    color = np.random.rand(3)
                    patch = torch.ones((3, patch_size[0], patch_size[1]), dtype=torch.float32) * torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
                    y[:, py:py + patch_size[0], px:px + patch_size[1]] = patch
                    break
            attempts += 1
    
    return y + (x * skin_mask)

# def color_block(x: torch.Tensor, amount: float) -> torch.Tensor:
#     if amount == 0.0:
#         return x
#     skin_mask = skin_segmentation(x)
#     background_mean = 1 - skin_mask.mean().item()  # Estimate of background area
#     if background_mean < 0.05:  # Only apply distortion if significant background is present
#         return x
#     x_new = x * ~skin_mask
#     amount = amount * background_mean
#     _, h, w = x_new.shape
#     patch_size = [int(max(32, min(h / 10 * amount, h))), int(max(32, min(w / 10 * amount, w)))]
    
#     y = x_new.clone()

#     # Convert numpy mask to torch tensor
#     mask_tensor = torch.from_numpy(~skin_mask).float()

#     h_max = h - patch_size[0]
#     w_max = w - patch_size[1]

#     num_patches = max(1, int(amount * 10))
#     attempts = 0
#     for _ in range(num_patches):
#         while True:
#             if h_max <= 0 or w_max <= 0 or attempts > 100:  # Avoid infinite loop by limiting attempts
#                 break
#             px = random.randint(0, w_max)
#             py = random.randint(0, h_max)
#             # Ensure the selected area is in the background
#             if px + patch_size[1] <= w and py + patch_size[0] <= h:
#                 patch_area = mask_tensor[py:py + patch_size[0], px:px + patch_size[1]]
#                 if torch.all(patch_area == 1):  # Ensure the entire patch is within the background
#                     color = np.random.rand(3)
#                     patch = torch.ones((3, patch_size[0], patch_size[1]), dtype=torch.float32) * torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
#                     y[:, py:py + patch_size[0], px:px + patch_size[1]] = patch
#                     break
#             attempts += 1
    
#     return y + (x * skin_mask)