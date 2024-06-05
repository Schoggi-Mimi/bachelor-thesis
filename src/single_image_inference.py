import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Any
import torch
from torchvision import transforms
from einops import rearrange
from utils.utils_data import (center_corners_crop, distort_images, map_distortion_values, resize_crop, discretization)

def test_single_image(
    img_path: str, 
    model: Optional[Any] = None, 
):
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
    
    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    
    # Image
    ax[0].imshow(np.array(original_img))
    #ax[0].set_title('Original Image', fontsize=15, fontweight='bold')
    ax[0].axis('off')

    # Radar chart
    ax[1] = plt.subplot(122, polar=True)
    ax[1].set_theta_offset(np.pi / 2)
    ax[1].set_theta_direction(-1)

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], criteria_names, fontsize=10.5, fontweight='bold')

    # Draw ylabels
    ax[1].set_rscale('linear')
    ax[1].set_ylim(0, 1)
    ax[1].set_yticklabels([])
    #ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1])
    #ax[1].set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=13, fontweight='bold')

    # Plot data
    values = scores.tolist()
    if len(values) != num_criteria:
        raise ValueError(f"Expected {num_criteria} scores, got {len(values)}")
    values += values[:1]  # Complete the loop
    ax[1].plot(angles, values, linewidth=4, linestyle='solid', label='Prediction', color='blue')
    ax[1].fill(angles, values, 'b', alpha=0.1)

    #plt.title('Distortion Severity Radar Chart', size=15, color='#C68642', y=1.1)
    #plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.show()
