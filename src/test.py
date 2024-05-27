import os
import random
from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from utils.utils_data import binarize_scores, get_features_scores
from utils.visualization import print_metrics, plot_all_confusion_matrices, plot_prediction_scores, plot_results
from data import BaseDataset

def load_images(image_paths):
    return [Image.open(path).convert("RGB") for path in image_paths]

def test(
    root: str = "SCIN",
    crop: bool = False,
    batch_size: int = 32,
    num_workers: int = 4,
    model: Optional[Any] = None,
    model_type: str = 'reg',
):
    assert model_type in ["reg", "cls", "mlp"], "phase must be in ['reg', 'cls', 'mlp']"
    embed_dir = os.path.join(root, "embeddings")
    feats_file = os.path.join(embed_dir, f"features.npy")
    scores_file = os.path.join(embed_dir, f"scores.npy")
    dataset = BaseDataset(root=root, crop=crop, phase="train", num_distortions=1)
    
    if os.path.exists(feats_file) and os.path.exists(scores_file):
        features = np.load(feats_file)
        scores = np.load(scores_file)
        print(f'Loaded features from {feats_file}')
        print(f'Loaded scores from {scores_file}')
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
        arniqa.eval().to(device)
        features, scores = get_features_scores(arniqa, dataloader, device, crop)
        if not os.path.exists(embed_dir):
            os.makedirs(embed_dir)
            
        np.save(feats_file, features)
        np.save(scores_file, scores)
        print(f'Saved features to {feats_file}')
        print(f'Saved scores to {scores_file}')
    
    image_indices = np.arange(len(dataset))
    if crop:
        image_indices = np.repeat(image_indices * 5, 5) + np.tile(np.arange(5), len(image_indices))
    test_features = features[image_indices]
    test_scores = scores[image_indices]
    if crop:
        test_scores = test_scores[::5]
        
    original_images = [Image.open(dataset.images[i]).convert("RGB") for i in np.unique(image_indices)]
    distorted_dir = os.path.join(root, "distorted")
    if os.path.exists(distorted_dir):
        distorted_image_paths = [os.path.join(distorted_dir, f"{i}.png") for i in np.unique(image_indices)]
        distorted_images = load_images(distorted_image_paths)
    else:
        distorted_images = None
    if model_type == 'reg':
        predictions = model.predict(test_features)
        if crop:
            predictions = np.reshape(predictions, (-1, 5, 7))  # Reshape to group crops per image
            predictions = np.mean(predictions, axis=1) # Average the predictions of the 5 crops of the same image
        predictions = np.clip(predictions, 0, 1)
        bin_pred = binarize_scores(predictions)
        bin_test = binarize_scores(test_scores)
        plot_prediction_scores(test_scores, predictions)
        plot_all_confusion_matrices(bin_test, bin_pred)
        print_metrics(bin_test, bin_pred)
        plot_results(original_images, distorted_images, predictions, test_scores)
    
    elif model_type == 'cls':
        test_scores = binarize_scores(test_scores)
        predictions = model.predict(test_features)
        plot_prediction_scores(test_scores, predictions)
        print_metrics(test_scores, predictions)
        plot_results(original_images, distorted_images, predictions, test_scores)

    elif model_type == 'mlp':
        predictions = model.predict(test_features)
        if crop:
            predictions = np.reshape(predictions, (-1, 5, 7))  # Reshape to group crops per image
            predictions = np.mean(predictions, axis=1) # Average the predictions of the 5 crops of the same image
        predictions = np.clip(predictions, 0, 1)
        bin_pred = binarize_scores(predictions)
        bin_test = binarize_scores(test_scores)
        plot_prediction_scores(test_scores, predictions)
        #plot_prediction_scores(bin_test, bin_pred)
        print_metrics(bin_test, bin_pred)
        plot_results(original_images, distorted_images, predictions, test_scores)