# arniqa_test.py
# Run: python arniqa_test.py --config_path config.yaml

import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser
from tqdm import tqdm
import yaml

from utils.utils_data import center_corners_crop

def process_images(root: str, regressor_dataset: str, output_csv: str):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                           regressor_dataset=regressor_dataset)
    model.eval().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'quality_score'])

        for filename in tqdm(os.listdir(root)):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)

                img = Image.open(img_path).convert("RGB")
                img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

                img = center_corners_crop(img, crop_size=224)
                img_ds = center_corners_crop(img_ds, crop_size=224)

                img = [transforms.ToTensor()(crop) for crop in img]
                img = torch.stack(img, dim=0)
                img = normalize(img).to(device)
                img_ds = [transforms.ToTensor()(crop) for crop in img_ds]
                img_ds = torch.stack(img_ds, dim=0)
                img_ds = normalize(img_ds).to(device)

                with torch.no_grad():
                    score = model(img, img_ds, return_embedding=False, scale_score=True)
                    score = np.round(score.mean(0).item(), 4)
                score = 1 - score
                writer.writerow([img_path, score])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config.yaml file")

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    process_images(config['arniqa_test']['root'], config['arniqa_test']['regressor_dataset'], config['arniqa_test']['output_csv'])