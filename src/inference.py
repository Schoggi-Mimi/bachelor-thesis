# inference.py
# Run: python inference.py --config_path config.yaml
import os
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse
import yaml

from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from utils.utils_data import resize_crop

class InferenceDataset(Dataset):
    def __init__(self, root: str = "images"):
        super().__init__()
        self._root = root
        self.crop_size = 224
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.images = [os.path.join(self._root, filename) for filename in os.listdir(self._root) if filename.endswith(('.png', '.jpg', 'jpeg'))]

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img_ds = resize_crop(img, crop_size=None, downscale_factor=2)

        img = resize_crop(img, crop_size=self.crop_size)
        img_ds = resize_crop(img_ds, crop_size=self.crop_size)
        img = transforms.ToTensor()(img)
        img_ds = transforms.ToTensor()(img_ds)

        img = self.normalize(img)
        img_ds = self.normalize(img_ds)
        return {"img": img, "img_ds": img_ds, "path": img_path}

    def __len__(self) -> int:
        return len(self.images)

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def get_features(arniqa, dataloader, device):
    feats = np.zeros((0, arniqa.encoder.feat_dim * 2))
    image_paths = []
    with tqdm(total=len(dataloader), desc="Extracting features", leave=False) as progress_bar:
        for batch in dataloader:
            img_orig = batch["img"].to(device)
            img_ds = batch["img_ds"].to(device)
            image_paths.extend(batch["path"])
            with torch.no_grad():
                _, f = arniqa(img_orig, img_ds, return_embedding=True)
            feats = np.concatenate((feats, f.cpu().numpy()), 0)
            progress_bar.update(1)
    return feats, image_paths

def make_predictions(model, features):
    predictions = model.predict(features)
    predictions = np.clip(predictions, 0, 1)
    return np.round(predictions, 4)

def save_predictions_to_csv(image_paths, predictions, output_csv):
    df = pd.DataFrame(predictions, columns=["background", "lighting", "focus", "orientation", "color_calibration", "resolution", "field_of_view"])
    df['image_path'] = image_paths
    df.to_csv(output_csv, index=False)

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
    arniqa.eval().to(device)
    model = load_model(config['inference']['model_path'])
    dataset = InferenceDataset(root=config['inference']['images_path'])
    dataloader = DataLoader(dataset, batch_size=config['inference']['batch_size'], shuffle=False, num_workers=config['inference']['num_workers'], pin_memory=True, drop_last=False)
    
    features, image_paths = get_features(arniqa, dataloader, device)

    if features.shape[0] == 0:
        print("No features were extracted. Exiting.")
        return
        
    predictions = make_predictions(model, features)
    save_predictions_to_csv(image_paths, predictions, config['inference']['csv_path'])
    
    print(f"Predictions saved to {config['inference']['csv_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for image quality assessment")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config.yaml file')

    args = parser.parse_args()

    main(config_path=args.config_path)