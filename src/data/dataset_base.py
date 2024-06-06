import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils_data import distort_images, map_distortion_values, resize_crop


class BaseDataset(Dataset):
    def __init__(self,
                 root: str = "images",
                 phase: str = "train",
                 num_distortions: int = 1):
        super().__init__()
        self._root = Path(root)
        self.phase = phase
        assert self.phase in ["train", "test"], "Phase must be 'train' or 'test'."
        self.num_distortions = num_distortions if phase == "train" else 1
        self.crop_size = 224
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = [os.path.join(self._root, filename) for filename in os.listdir(self._root) if filename.endswith(('.png', '.jpg', 'jpeg'))]
            
    def __getitem__(self, idx: int) -> dict:
        img_idx = idx // self.num_distortions
        img_path = self.images[img_idx]
        img = Image.open(img_path).convert("RGB")
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        if self.phase == "train":
            img = transforms.ToTensor()(img)
            img_ds = transforms.ToTensor()(img_ds)
            img, distort_functions, distort_values = distort_images(img)
            img_ds, _, _ = distort_images(img_ds, distort_functions=distort_functions, distort_values=distort_values)
            mapped_values = map_distortion_values(distort_functions, distort_values)
            img = transforms.ToPILImage()(img)
            img_ds = transforms.ToPILImage()(img_ds)

            if not os.path.exists(os.path.join(self._root, 'distorted')):
                os.makedirs(os.path.join(self._root, 'distorted'))
            img_path = os.path.basename(img_path)
            path = os.path.join(self._root, f"distorted/{img_path}")
            #img.save(path)

        elif self.phase == "test":
            mapped_values = torch.tensor([0.0] * 7, dtype=torch.float32)  # Placeholder values, actual scores are loaded separately

        img = resize_crop(img, crop_size=self.crop_size)
        img_ds = resize_crop(img_ds, crop_size=self.crop_size)
        img = transforms.ToTensor()(img)
        img_ds = transforms.ToTensor()(img_ds)

        img = self.normalize(img)
        img_ds = self.normalize(img_ds)
        return {"img": img, "img_ds": img_ds, "label": mapped_values}

    def __len__(self) -> int:
        return len(self.images) * self.num_distortions