import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Any
import torch
from torchvision import transforms
from einops import rearrange
from utils.utils_data import (center_corners_crop, distort_images, map_distortion_values, resize_crop, binarize_scores)

def test_single_image(
    img_path: str, 
    crop: bool = False, 
    model: Optional[Any] = None, 
    model_type: str = 'reg',
):
    assert model_type in ["reg", "cls", "mlp"], "phase must be in ['reg', 'cls', 'mlp']"
    img, img_ds = preprocess_image(img_path, crop)
    predictions = predict_distortion_scores(img, img_ds, model, model_type, crop)
    original_img = Image.open(img_path)
    plot_results(original_img, predictions)

def preprocess_image(img_path, crop):
    img = Image.open(img_path).convert("RGB")
    img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

    if crop:
        crops = center_corners_crop(img, crop_size=crop_size)
        crops = [transforms.ToTensor()(crop) for crop in crops]
        img = torch.stack(crops, dim=0)

        crops_ds = center_corners_crop(img_ds, crop_size=crop_size)
        crops_ds = [transforms.ToTensor()(crop) for crop in crops_ds]
        img_ds = torch.stack(crops_ds, dim=0)
    else:
        img = resize_crop(img, crop_size=224)
        img_ds = resize_crop(img_ds, crop_size=224)
        img = transforms.ToTensor()(img)
        img_ds = transforms.ToTensor()(img_ds)
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    img_ds = normalize(img_ds)
    return img, img_ds

def predict_distortion_scores(img, img_ds, model, model_type, crop):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
    arniqa.eval().to(device)
    feats = np.zeros((0, arniqa.encoder.feat_dim * 2))
    img = img.unsqueeze(0).to(device)
    img_ds = img_ds.unsqueeze(0).to(device)
    if crop:
        img = rearrange(img, "b n c h w -> (b n) c h w")
        img_ds = rearrange(img_ds, "b n c h w -> (b n) c h w")
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, f = arniqa(img, img_ds, return_embedding=True)
    features = np.concatenate((feats, f.cpu().numpy()), 0)
    #features = features.cpu().numpy()
    predictions = model.predict(features)
    if crop:
        predictions = np.mean(predictions, axis=0) # Average the predictions of the 5 crops of the same image
    if model_type == 'reg' or model_type == 'mlp':
        predictions = np.clip(predictions, 0, 1)
        predictions = binarize_scores(predictions)
    return predictions.squeeze(0)

def plot_results(original_img, scores):
    criteria_names = ["Background", "Lighting", "Focus", "Orientation", "Color calibration", "Resolution", "Field of view"]
    num_criteria = len(criteria_names)
    
    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Image
    ax[0].imshow(np.array(original_img))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Radar chart
    ax[1] = plt.subplot(122, polar=True)
    ax[1].set_theta_offset(np.pi / 2)
    ax[1].set_theta_direction(-1)

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], criteria_names)

    # Draw ylabels
    ax[1].set_rscale('linear')
    ax[1].set_ylim(0, 4)
    ax[1].set_yticks([0, 1, 2, 3, 4])
    ax[1].set_yticklabels(['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4'])

    # Plot data
    values = scores.tolist()
    if len(values) != num_criteria:
        raise ValueError(f"Expected {num_criteria} scores, got {len(values)}")
    values += values[:1]  # Complete the loop
    ax[1].plot(angles, values, linewidth=2, linestyle='solid', label='Prediction')
    ax[1].fill(angles, values, 'b', alpha=0.1)

    plt.title('Distortion Severity Radar Chart', size=15, color='#C68642', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.show()

def center_corners_crop_old(img, crop_size):
    width, height = img.size
    center_x, center_y = width // 2, height // 2
    crop_width, crop_height = crop_size, crop_size

    crop_boxes = [
        (center_x - crop_width // 2, center_y - crop_height // 2, center_x + crop_width // 2, center_y + crop_height // 2)
    ]

    crops = [img.crop(box) for box in crop_boxes]
    return crops