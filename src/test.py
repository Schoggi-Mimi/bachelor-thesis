# test.py
# Run: python test.py --config_path config.yaml

import argparse
import json
import os
import pickle
from typing import Any, Optional

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from data import BaseDataset
from utils.utils_data import get_features_scores
from utils.visualization import plot_results


def test(
    root: str,
    batch_size: int,
    num_workers: int,
    model: Optional[Any],
    data_type: str
):
    image_paths = [os.path.join(root, filename) for filename in os.listdir(root) if filename.endswith(('.png', '.jpg', 'jpeg'))]
    original_images = [Image.open(path).convert("RGB") for path in image_paths]

    embed_dir = os.path.join(root, "embeddings")
    feats_file = os.path.join(embed_dir, f"features.npy")
    scores_file = os.path.join(embed_dir, f"scores.npy")
    dataset = BaseDataset(root=root, phase="test", num_distortions=1)

    if os.path.exists(feats_file):
        features = np.load(feats_file)
        print(f'Loaded features from {feats_file}')
        if data_type == 's':
            test_scores = np.load(scores_file)
            print(f'Loaded scores from {scores_file}')
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
        arniqa.eval().to(device)
        features, test_scores = get_features_scores(arniqa, dataloader, device)
        if not os.path.exists(embed_dir):
            os.makedirs(embed_dir)
        np.save(feats_file, features)
        np.save(scores_file, test_scores)
        print(f'Saved features to {feats_file}')
        print(f'Saved scores to {scores_file}')

    if data_type == 'a':
        with open(os.path.join(root, "scores.json"), "r") as json_file:
            scores_data = json.load(json_file)
            
        criteria_order = ['background', 'lighting', 'focus', 'orientation', 'color_calibration', 'resolution', 'field_of_view']
        test_scores = []
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            test_scores.append([scores_data.get(filename, {}).get(key, 0.0) for key in criteria_order])
        test_scores = np.array(test_scores, dtype=np.float32)
        distorted_images = None
    
    elif data_type == 's':
        distorted_dir = os.path.join(root, "distorted")
        distorted_image_paths = [i for i in image_paths]
        distorted_images = [Image.open(os.path.join(distorted_dir, os.path.basename(path))).convert("RGB") for path in distorted_image_paths]

    predictions = model.predict(features)
    predictions = np.clip(predictions, 0, 1)
    plot_results(original_images, distorted_images, predictions, predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for image quality assessment")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config.yaml file')

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    with open(config['test']['model_path'], 'rb') as file:
        model = pickle.load(file)

    test(
        root=config['test']['root'],
        batch_size=config['test']['batch_size'],
        num_workers=config['test']['num_workers'],
        model=model,
        data_type=config['test']['data_type']
    )