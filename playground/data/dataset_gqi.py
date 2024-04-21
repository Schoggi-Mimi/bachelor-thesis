import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils_data import (center_corners_crop, distort_images,
                              map_distortion_values, resize_crop)


class GQIDataset(Dataset):
    def __init__(self,
                 root: str = "images",
                 crop: bool = True,
                 normalize: bool = True,):
        super().__init__()
        self._root = Path(root)
        self.target_size = 512
        self.crop_size = 224
        self.to_crop = crop
        self.to_normalize = normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = [os.path.join(self._root, filename) for filename in os.listdir(self._root) if filename.endswith('.png')]

    def __getitem__(self, index: int) -> dict:
        img = Image.open(self.images[index]).convert("RGB")
        
        width, height = img.size
        aspect_ratio = width / height
        if width < height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        
        img = img.resize((new_width, new_height), Image.BICUBIC)
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)
        img = transforms.ToTensor()(img)
        img_ds = transforms.ToTensor()(img_ds)

        img, distort_functions, distort_values = distort_images(img)
        img_ds, _, _ = distort_images(img_ds, distort_functions=distort_functions, distort_values=distort_values)

        if self.to_crop:
            img = transforms.ToPILImage()(img)
            img_ds = transforms.ToPILImage()(img_ds)

            crops = center_corners_crop(img, crop_size=self.crop_size)
            crops = [transforms.ToTensor()(crop) for crop in crops]
            img = torch.stack(crops, dim=0)

            crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
            crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
            img_ds = torch.stack(crops_ds, dim=0)

        if self.to_normalize:
            img = self.normalize(img)
            img_ds = self.normalize(img_ds)
        mapped_values = map_distortion_values(distort_functions, distort_values)

        return {"img": img, "img_ds": img_ds, "label": mapped_values}

    def __len__(self) -> int:
        return len(self.images)