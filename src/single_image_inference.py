import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Any
import torch
from torchvision import transforms

def single_image_inference(img_path: str, model: torch.nn.Module = None, xgb: Optional[Any] = None, crop_size: int = 224, regression: bool = False):
    img, img_ds = preprocess_image(img_path, crop_size)
    scores = predict_distortion_scores(model, xgb, img, img_ds, regression)
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

def predict_distortion_scores(model, xgb, img, img_ds, regression):
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, features = model(img, img_ds, return_embedding=True)
    features = features.cpu().numpy()
    predictions = xgb.predict(features)
    predictions = np.mean(predictions, axis=0) # Average the predictions of the 5 crops of the same image
    if regression:
        predictions = np.clip(predictions, 0, 1)
    return predictions

def plot_results(original_img, scores):
    criteria_names = ["Background", "Lighting", "Focus", "Orientation", "Color calibration", "Resolution", "Field of view"]
    num_criteria = len(criteria_names)
    
    # Setup radar chart
    angles = np.linspace(0, 2 * np.pi, num_criteria, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the image
    ax[0].imshow(np.array(original_img))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Create the radar chart
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
    values += values[:1]  # Complete the loop
    ax[1].plot(angles, values, linewidth=2, linestyle='solid', label='Prediction')
    ax[1].fill(angles, values, 'b', alpha=0.1)

    plt.title('Distortion Severity Radar Chart', size=15, color='#C68642', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.show()

def center_corners_crop(img, crop_size):
    width, height = img.size
    center_x, center_y = width // 2, height // 2
    crop_width, crop_height = crop_size, crop_size

    crop_boxes = [
        (center_x - crop_width // 2, center_y - crop_height // 2, center_x + crop_width // 2, center_y + crop_height // 2)
    ]

    crops = [img.crop(box) for box in crop_boxes]
    return crops