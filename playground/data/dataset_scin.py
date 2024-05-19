from PIL import Image
import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

from data.dataset_base_iqa import IQADataset
from utils.utils_data import resize_crop, center_corners_crop

class SCINDataset(IQADataset):
    """
    Dermatology dataset class.

    Args:
        root (string): root directory of the dataset
        phase (string): indicates the phase of the dataset. Value must be in ['train', 'test', 'val', 'all']. Default is 'train'.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): the center crop and the 4 corners of the image (5 x 3 x crop_size x crop_size)
            img_ds (Tensor): downsampled version of the image (scale factor 2)
    """
    def __init__(self,
                 root: str,
                 phase: str = "train",
                 crop_size: int = 224,
                 split_idx: int = 0):
        mos_type = "mos"
        mos_range = (1, 5)
        is_synthetic = False
        super().__init__(root, mos_type=mos_type, mos_range=mos_range, is_synthetic=is_synthetic, phase=phase, split_idx=split_idx, crop_size=crop_size)

        self._path = os.path.join(root, "images")
        self.target_size = 512
        self.images = [os.path.join(self._path, filename) for filename in os.listdir(self._path) if filename.endswith('.jpg')]

        scores_csv = pd.read_csv(os.path.join(self.root, "scin_median_confidence.csv"))
        self.images = scores_csv["image_path"].values.tolist()
        self.images = np.array([os.path.join(self.root, el) for el in self.images])

        self.mos = np.array(scores_csv["min_max_confidence"].values.tolist())

        if self.phase != "all":
            split_idxs = np.load(self.root / "splits" / f"{self.phase}.npy")[self.split_idx]
            split_idxs = np.array(list(filter(lambda x: x != -1, split_idxs)))  # Remove the padding (i.e. -1 indexes)
            self.images = self.images[split_idxs]
            self.mos = self.mos[split_idxs]

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
        
        #img_ds = img.resize((img.width // 2, img.height // 2), Image.BICUBIC)
        img = img.resize((new_width, new_height), Image.BICUBIC)
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        crops = center_corners_crop(img, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]
        img = torch.stack(crops, dim=0)

        crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        img_ds = torch.stack(crops_ds, dim=0)

        img = self.normalize(img)
        img_ds = self.normalize(img_ds)
        mos = self.mos[index]
        return {"img": img, "img_ds": img_ds, "mos": mos}

    def __len__(self) -> int:
        return len(self.images)