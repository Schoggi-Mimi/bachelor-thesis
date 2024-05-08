from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.neural_network import MLPRegressor
from torchvision import transforms

from utils.utils_data import \
    center_corners_crop  # Make sure this is defined or imported


def single_image_inference(img_path: str, model: torch.nn.Module = None, regressor: Optional[MLPRegressor] = None, crop_size: int = 224):
    img, img_ds = preprocess_image(img_path, crop_size)
    scores = predict_distortion_scores(model, regressor, img, img_ds)
    original_img = Image.open(img_path)
    plot_results(original_img, scores)

def preprocess_image(img_path, crop_size):
    img = Image.open(img_path).convert("RGB")
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

    crops = center_corners_crop(img, crop_size=crop_size)
    crops = [transforms.ToTensor()(crop) for crop in crops]
    img = torch.stack(crops, dim=0)

    crops_ds = center_corners_crop(img_ds, crop_size=crop_size)
    crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
    img_ds = torch.stack(crops_ds, dim=0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    img_ds = normalize(img_ds)
    return img, img_ds

def predict_distortion_scores(model, regressor, img, img_ds):
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, features = model(img, img_ds, return_embedding=True)
    features = features.cpu().numpy()
    predictions = regressor.predict(features)
    predictions = np.mean(predictions, axis=0) # Average the predictions of the 5 crops of the same image
    predictions = np.clip(predictions, 0, 1)

    return predictions

def plot_results(original_img, scores):
    criteria_names = ['Lighting', 'Focus', 'Orientation', 'Color Calibration', 'Background', 'Resolution', 'Field of View']
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image
    ax[0].imshow(np.array(original_img))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Plot the distortion scores
    ax[1].barh(criteria_names, scores, color='blue')
    ax[1].set_xlim(0, 1)
    ax[1].set_title('Distortion Severity')
    ax[1].set_xlabel('Severity Scale (0 - No distortion, 1 - Maximum distortion)')

    plt.tight_layout()
    plt.show()
