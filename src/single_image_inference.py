# single_image_inference.py
# Run: python single_image_inference.py --config_path config.yaml
import argparse
import pickle
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from utils.utils_data import resize_crop


def test_single_image(img_path: str, model: Optional[Any] = None):
    img, img_ds = preprocess_image(img_path)
    predictions = predict_distortion_scores(img, img_ds, model)
    original_img = Image.open(img_path)
    plot_results(original_img, predictions)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

    img = resize_crop(img, crop_size=224)
    img_ds = resize_crop(img_ds, crop_size=224)
    img = transforms.ToTensor()(img)
    img_ds = transforms.ToTensor()(img_ds)
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    img_ds = normalize(img_ds)
    return img, img_ds

def predict_distortion_scores(img, img_ds, model):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
    arniqa.eval().to(device)
    feats = np.zeros((0, arniqa.encoder.feat_dim * 2))
    img = img.unsqueeze(0).to(device)
    img_ds = img_ds.unsqueeze(0).to(device)

    with torch.no_grad():
        _, f = arniqa(img, img_ds, return_embedding=True)
    features = np.concatenate((feats, f.cpu().numpy()), 0)
    predictions = model.predict(features)
    predictions = np.clip(predictions, 0, 1)
    return predictions.squeeze(0)

def plot_results(original_img, scores):
    criteria_names = ["Background", "Lighting", "Focus", "Orientation", "Color calibration", "Resolution", "Field of view"]
    num_criteria = len(criteria_names)
    
    angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(np.array(original_img))
    ax[0].axis('off')

    ax[1] = plt.subplot(122, polar=True)
    ax[1].set_theta_offset(np.pi / 2)
    ax[1].set_theta_direction(-1)

    plt.xticks(angles[:-1], criteria_names, fontsize=10.5, fontweight='bold')

    ax[1].set_rscale('linear')
    ax[1].set_ylim(0, 1)
    ax[1].set_yticklabels([])

    values = scores.tolist()
    if len(values) != num_criteria:
        raise ValueError(f"Expected {num_criteria} scores, got {len(values)}")
    values += values[:1]
    ax[1].plot(angles, values, linewidth=4, linestyle='solid', label='Prediction', color='blue')
    ax[1].fill(angles, values, 'b', alpha=0.1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single image inference script for image quality assessment")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config.yaml file')

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    with open(config['single_image_inference']['model_path'], 'rb') as file:
        model = pickle.load(file)

    test_single_image(config['single_image_inference']['image_path'], model)