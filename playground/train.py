import os
from pathlib import Path
#from dotmap import DotMap
from typing import Optional, Tuple

import numpy as np
import torch
from einops import rearrange
#import wandb
#from wandb.wandb_run import Run
from PIL import ImageFile
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GQIDataset


def train_regressor(
        root: str = "images",
        crop: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        model: torch.nn.Module = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):
    # save checkpoints
    dataset = GQIDataset(root=root, crop=crop, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    features, scores = get_features_scores(model, dataloader, device)

    train_indices = np.arange(len(dataset))
    if crop:
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
    train_features = features[train_indices]
    train_scores = scores[train_indices]

    X_train, X_test, y_train, y_test = train_test_split(train_features, train_scores, test_size=0.2, random_state=42)
    regressor = LinearRegression().fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    return mae

def get_features_scores(model: torch.nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.zeros((0, model.encoder.feat_dim * 2))   # Double the features because of the original and downsampled image
    scores = np.zeros((0, 4))
    total_iterations = len(dataloader) * dataloader.batch_size
    with tqdm(total=total_iterations, desc="Extracting features", leave=False) as progress_bar:
        for _, batch in enumerate(dataloader):
            img_orig = batch["img"].to(device)
            img_ds = batch["img_ds"].to(device)
            label = batch["label"]
    
            img_orig = rearrange(img_orig, "b n c h w -> (b n) c h w")
            img_ds = rearrange(img_ds, "b n c h w -> (b n) c h w")
            label = torch.cat([label.repeat(5, 1) for label in batch["label"]], dim=0) # repeat label for each crop

    
            with torch.cuda.amp.autocast(), torch.no_grad():
                _, f = model(img_orig, img_ds, return_embedding=True)
    
            feats = np.concatenate((feats, f.cpu().numpy()), 0)
            scores = np.concatenate((scores, label.numpy()), 0)
            progress_bar.update(1)
    return feats, scores