import os
from pathlib import Path
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils_data import (center_corners_crop, distort_images,
                              map_distortion_values, resize_crop)


class GQIDataset(Dataset):
    def __init__(self,
                 root: str = "images",
                 crop: bool = True,
                 normalize: bool = True,
                 phase: str = "train"
                ):
        super().__init__()
        self._root = Path(root)
        self.phase = phase
        assert self.phase in ["train", "test"], "phase must be in 'train' or 'test'"
        self.target_size = 512
        self.crop_size = 224
        self.to_crop = crop
        self.to_normalize = normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = [os.path.join(self._root, filename) for filename in os.listdir(self._root) if filename.endswith('.png')]

        if self.phase == "test":
            with open(os.path.join(root, "scores.json"), "r") as json_file:
                self.labels = json.load(json_file)
            self.criteria_order = ['lighting', 'focus', 'orientation', 'color_calibration', 'background', 'resolution', 'field_of_view']
            
    def __getitem__(self, index: int) -> dict:
        img_path = self.images[index]
        img = Image.open(img_path).convert("RGB")
        
        #width, height = img.size
        #aspect_ratio = width / height
        #if width < height:
            #new_width = self.target_size
            #new_height = int(self.target_size / aspect_ratio)
        #else:
            #new_height = self.target_size
            #new_width = int(self.target_size * aspect_ratio)
        
        #img = img.resize((new_width, new_height), Image.BICUBIC)
        #img = resize_crop(img_A, crop_size=self.crop_size)
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        if self.phase == "train":
            img = transforms.ToTensor()(img)
            img_ds = transforms.ToTensor()(img_ds)
            img, distort_functions, distort_values = distort_images(img)
            img_ds, _, _ = distort_images(img_ds, distort_functions=distort_functions, distort_values=distort_values)
            mapped_values = map_distortion_values(distort_functions, distort_values)
            img = transforms.ToPILImage()(img)
            img_ds = transforms.ToPILImage()(img_ds)

        elif self.phase == "test":
            filename = os.path.basename(img_path)
            label_values = self.labels.get(filename, {})
            mapped_values = torch.tensor([label_values.get(key, 0.0) for key in self.criteria_order], dtype=torch.float32)

        if self.to_crop:
            crops = center_corners_crop(img, crop_size=self.crop_size)
            crops = [transforms.ToTensor()(crop) for crop in crops]
            img = torch.stack(crops, dim=0)

            crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
            crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
            img_ds = torch.stack(crops_ds, dim=0)

        if self.to_normalize:
            img = self.normalize(img)
            img_ds = self.normalize(img_ds)
            
        return {"img": img, "img_ds": img_ds, "label": mapped_values}

    def __len__(self) -> int:
        return len(self.images)