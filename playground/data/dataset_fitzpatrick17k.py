from PIL import Image
import os
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset

# from data.dataset_base_iqa import IQADataset
from utils.utils_data import resize_crop, center_corners_crop

class F17KDataset(Dataset):
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
                 crop_size: int = 224):
        #super().__init__(root, phase=phase, crop_size=crop_size)
        
        self.root = root # f17k
        #self.phase = phase # train, val, test
        self._path = os.path.join(root, phase)
        #assert self.phase in ["train", "test", "val", "all"], "phase must be in ['train', 'test', 'val', 'all']"
        self.crop_size = crop_size
        self.target_size = 512
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = [os.path.join(self._path, filename) for filename in os.listdir(self._path) if filename.endswith('.jpg')]
        #self.images = [os.path.join(self.root, filename) for filename in os.listdir(self.root) if filename.endswith('.jpg')]

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
        
        img_ds = img.resize((img.width // 2, img.height // 2), Image.BICUBIC)
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        crops = center_corners_crop(img, crop_size=self.crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]
        img = torch.stack(crops, dim=0)

        crops_ds = center_corners_crop(img_ds, crop_size=self.crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        img_ds = torch.stack(crops_ds, dim=0)

        img = self.normalize(img)
        img_ds = self.normalize(img_ds)
        return {"img": img, "img_ds": img_ds}

    def __len__(self) -> int:
        return len(self.images)

class SCINDataset(Dataset):
    """
    Dermatology dataset class.

    Args:
        root (string): root directory of the dataset
        path (string): indicates the path of the dataset. Default is 'images'.
        crop_size (int): size of each crop. Default is 224.

    Returns:
        dictionary with keys:
            img (Tensor): the center crop and the 4 corners of the image (5 x 3 x crop_size x crop_size)
            img_ds (Tensor): downsampled version of the image (scale factor 2)
            mos (float): mean opinion score of the image (in range [1, 5])
    """
    def __init__(self,
                 root: str,
                 path: str = "images",
                 crop_size: int = 224):

        mos_range = (1, 5)
        self.root = root # SCIN
        self._path = os.path.join(root, path)
        self.crop_size = crop_size
        self.target_size = 512
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = [os.path.join(self._path, filename) for filename in os.listdir(self._path) if filename.endswith('.jpg')]

        scores_csv = pd.read_csv(os.path.join(self.root, "scin_median_confidence.csv"))
        self.images = scores_csv["image_path"].values.tolist()
        self.images = np.array([os.path.join(self.root, el) for el in self.images])

        self.mos = np.array(scores_csv["min_max_confidence"].values.tolist())

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