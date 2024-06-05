import os
import random
import numpy as np
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List

from utils.utils_data import resize_crop


class InferenceDataset(Dataset):
    def __init__(self,
                 root: str = "images"):
        super().__init__()
        self._root = Path(root)
        self.crop_size = 224
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = [os.path.join(self._root, filename) for filename in os.listdir(self._root) if filename.endswith(('.png', '.jpg', 'jpeg'))]
            
    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        img = resize_crop(img, crop_size=self.crop_size)
        img_ds = resize_crop(img_ds, crop_size=self.crop_size)
        img = transforms.ToTensor()(img)
        img_ds = transforms.ToTensor()(img_ds)

        img = self.normalize(img)
        img_ds = self.normalize(img_ds)
        return {"img": img, "img_ds": img_ds, "path": img_path}

    def __len__(self) -> int:
        return len(self.images)