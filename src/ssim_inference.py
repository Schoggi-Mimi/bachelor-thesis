# ssim_inference.py
# Run: python ssim_inference.py --config_path config.yaml
import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
from tqdm import tqdm
import argparse
import yaml

from utils.utils_data import resize_crop

def calculate_ssim(original_img, distorted_img):
    if original_img.size != distorted_img.size:
        original_img = resize_crop(original_img, crop_size=224)
        distorted_img = resize_crop(distorted_img, crop_size=224)
    original_img_np = np.array(original_img)
    distorted_img_np = np.array(distorted_img)
    
    if original_img_np.shape[0] < 7 or original_img_np.shape[1] < 7:
        raise ValueError("Image dimensions must be at least 7x7 pixels.")
    
    ssim_value, _ = compare_ssim(original_img_np, distorted_img_np, multichannel=True, channel_axis=2, full=True)
    return ssim_value

def load_images(original_path, distorted_path):
    original_images = sorted([os.path.join(original_path, f) for f in os.listdir(original_path) if f.endswith(('.png', '.jpg', 'jpeg'))])
    distorted_images = sorted([os.path.join(distorted_path, f) for f in os.listdir(distorted_path) if f.endswith(('.png', '.jpg', 'jpeg'))])
    return original_images, distorted_images

def save_scores_to_csv(image_paths, ssim_scores, output_csv):
    df = pd.DataFrame({
        "original_image_path": image_paths[0],
        "distorted_image_path": image_paths[1],
        "SSIM": ssim_scores
    })
    df.to_csv(output_csv, index=False)

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    original_images, distorted_images = load_images(config['ssim_inference']['original_path'], config['ssim_inference']['distorted_path'])
    
    if len(original_images) != len(distorted_images):
        print("The number of original and distorted images must be the same.")
        return

    ssim_scores = []
    image_paths = ([], [])

    with tqdm(total=len(original_images), desc="Calculating SSIM", leave=False) as progress_bar:
        for orig_path, dist_path in zip(original_images, distorted_images):
            orig_img = Image.open(orig_path).convert("RGB")
            dist_img = Image.open(dist_path).convert("RGB")
            
            ssim = calculate_ssim(orig_img, dist_img)
            ssim_scores.append(ssim)
            image_paths[0].append(orig_path)
            image_paths[1].append(dist_path)
            
            progress_bar.update(1)
            
    inverted_ssim_values = [1 - s for s in ssim_scores]
    save_scores_to_csv(image_paths, inverted_ssim_values, config['ssim_inference']['csv_path'])
    print(f"SSIM scores saved to {config['ssim_inference']['csv_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSIM calculation script for image quality assessment")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config.yaml file')

    args = parser.parse_args()

    main(config_path=args.config_path)