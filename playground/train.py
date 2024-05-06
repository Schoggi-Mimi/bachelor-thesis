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
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from data import GQIDataset


def train_regressor(
    root: str = "images",
    crop: bool = True,
    normalize: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    model: torch.nn.Module = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    cross_val: bool = False,
):
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    dataset = GQIDataset(root=root, crop=crop, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    features, scores = get_features_scores(model, dataloader, device)

    image_indices = np.arange(len(dataset))
    train_img_indices, val_img_indices = train_test_split(image_indices, test_size=0.25, random_state=42, shuffle=False)
    train_indices = np.repeat(train_img_indices * 5, 5) + np.tile(np.arange(5), len(train_img_indices))
    val_indices = np.repeat(val_img_indices * 5, 5) + np.tile(np.arange(5), len(val_img_indices))

    train_features = features[train_indices]
    train_scores = scores[train_indices]
    val_features = features[val_indices]
    val_scores = scores[val_indices]
    val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one
    orig_val_indices = val_indices[::5] // 5  # Get original indices
    
    regressor = LinearRegression().fit(train_features, train_scores)
    #regressor = RandomForestRegressor(n_estimators=50, random_state=42).fit(train_features, train_scores)
    if cross_val:
        predictions = cross_val_score(regressor, val_features, val_scores, cv=5, scoring='neg_mean_squared_error')
        predictions = -predictions
        print("Cross-validation MSE scores:", predictions)
        print("Average CV MSE:", np.mean(predictions))
    else:
        predictions = regressor.predict(val_features)
        predictions = np.mean(np.reshape(predictions, (-1, 5, 7)), axis=1)  # Average the predictions of the 5 crops of the same image
        #srocc_dataset["global"].append(stats.spearmanr(predictions, val_scores)[0])
        #plcc_dataset["global"].append(stats.pearsonr(predictions, val_scores)[0])
        mae = mean_absolute_error(val_scores, predictions)
        print("Mean Absolute Error:", mae)
        mse = mean_squared_error(val_scores, predictions)
        print("Mean Squarred Error:", mse)

    distortion_criteria = ["lighting", "focus", "orientation", "color_calibration", "background", "resolution", "field_of_view"]
    
    fig, axes = plt.subplots(1, predictions.shape[1], figsize=(20, 4), sharey=True)
    for i, ax in enumerate(axes):
        ax.scatter(range(len(val_scores[:, i])), val_scores[:, i], color='blue', label='Actual Scores')
        ax.scatter(range(len(predictions[:, i])), predictions[:, i], color='red', label='Predicted Scores', alpha=0.5)
        ax.set_title(f'Score Metric {i+1} ({distortion_criteria[i]})')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Scores')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

    return regressor

def get_features_scores(model: torch.nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.zeros((0, model.encoder.feat_dim * 2))   # Double the features because of the original and downsampled image
    scores = np.zeros((0, 7))
    #total_iterations = len(dataloader) * dataloader.batch_size
    with tqdm(total=len(dataloader), desc="Extracting features", leave=False) as progress_bar:
        for _, batch in enumerate(dataloader):
            img_orig = batch["img"].to(device)
            img_ds = batch["img_ds"].to(device)
            label = batch["label"].repeat(5, 1).to(device)
    
            img_orig = rearrange(img_orig, "b n c h w -> (b n) c h w")
            img_ds = rearrange(img_ds, "b n c h w -> (b n) c h w")
            #label = torch.cat([label.repeat(5, 1) for label in batch["label"]], dim=0) # repeat label for each crop

    
            with torch.cuda.amp.autocast(), torch.no_grad():
                _, f = model(img_orig, img_ds, return_embedding=True)
    
            feats = np.concatenate((feats, f.cpu().numpy()), 0)
            scores = np.concatenate((scores, label.numpy()), 0)
            progress_bar.update(1)
    return feats, scores