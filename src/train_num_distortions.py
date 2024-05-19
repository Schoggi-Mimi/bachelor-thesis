import os
import random
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from xgboost import XGBRegressor, XGBClassifier
import wandb
from wandb.integration.xgboost import WandbCallback
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from models.multioutput_xgb import MultiOutputXGBClassifier, train_xgbclassifier, sweep_train
from utils.utils_data import binarize_scores
from data import BaseDataset

def train():
    wandb.init(reinit=True)
    config = wandb.config
    dataset = BaseDataset(root=config.root, crop=config.crop, phase="train", num_distortions=config.num_distortions)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
    model.eval().to(device)
    features, scores = get_features_scores(model, dataloader, device, config.root, config.crop, config.num_distortions)
    
    image_indices = np.arange(len(dataset))
    train_img_indices, val_img_indices = train_test_split(image_indices, test_size=0.25, random_state=42, shuffle=True)
    train_features = features[train_img_indices]
    train_scores = scores[train_img_indices]
    val_features = features[val_img_indices]
    val_scores = scores[val_img_indices]
    train_scores = binarize_scores(train_scores)
    val_scores = binarize_scores(val_scores)

    params = {
        'booster': 'gbtree',
        'n_estimators': config.n_estimators,
        'max_depth': config.max_depth,
        'learning_rate': config.learning_rate,
        'subsample': config.subsample,
        'objective': 'multi:softprob',
        'random_state': 42,
        'eval_metric': ['mlogloss', 'merror', 'auc'],
        'early_stopping_rounds': 10,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'tree_method': 'hist',
        'device': 'cpu'
    }
    classifier = MultiOutputXGBClassifier(params=params, num_class=5)
    classifier.fit(train_features, train_scores, eval_set=(val_features, val_scores))
    predictions = classifier.predict(val_features)

    overall_srocc = stats.spearmanr(predictions.flatten(), val_scores.flatten())[0]
    criteria_sroccs = [stats.spearmanr(predictions[:, i], val_scores[:, i])[0] for i in range(predictions.shape[1])]
    wandb.log({"overall_srocc": overall_srocc, "criteria_sroccs": criteria_sroccs})
    #wandb.finish()
    plot_all_confusion_matrices(val_scores, predictions)
    print_metrics(val_scores, predictions)

def get_features_scores(model: torch.nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        save_dir: str,
                        crop: bool, 
                        num_distortions: int,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    embed_dir = os.path.join(save_dir, "embeddings")
    feats_file = os.path.join(embed_dir, f"features_{num_distortions}.npy")
    scores_file = os.path.join(embed_dir, f"scores_{num_distortions}.npy")
    
    if os.path.exists(feats_file) and os.path.exists(scores_file):
        feats = np.load(feats_file)
        scores = np.load(scores_file)
        print(f'Loaded features from {feats_file}')
        print(f'Loaded scores from {scores_file}')
        return feats, scores
        
    feats = np.zeros((0, model.encoder.feat_dim * 2))   # Double the features because of the original and downsampled image (0, 4096)
    scores = np.zeros((0, 7))

    with tqdm(total=len(dataloader), desc="Extracting features", leave=False) as progress_bar:
        for _, batch in enumerate(dataloader):
            img_orig = batch["img"].to(device)
            img_ds = batch["img_ds"].to(device)

            if crop:
                label = batch["label"].repeat(5, 1) # repeat label for each crop
                img_orig = rearrange(img_orig, "b n c h w -> (b n) c h w")
                img_ds = rearrange(img_ds, "b n c h w -> (b n) c h w")
            elif crop is False:
                label = batch["label"]

            with torch.cuda.amp.autocast(), torch.no_grad():
                _, f = model(img_orig, img_ds, return_embedding=True)
    
            feats = np.concatenate((feats, f.cpu().numpy()), 0)
            scores = np.concatenate((scores, label.numpy()), 0)
            progress_bar.update(1)
    
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
        
    np.save(feats_file, feats)
    np.save(scores_file, scores)
    print(f'Saved features to {feats_file}')
    print(f'Saved scores to {scores_file}')
    
    return feats, scores

def print_metrics(val, pred):
    criteria = ['Background', 'Lighting', 'Focus', 'Orientation', 'Color calibration', 'Resolution', 'Field of view']
    print(f"\n{'Criteria':^18} | {'Precision (%)':^14} | {'Recall (%)':^12} | {'PLCC':^10} | {'SROCC':^10} |")
    print("------------------------------------------------------------------------------")

    for i in range(pred.shape[1]):
        mae_value = mean_absolute_error(val[:, i], pred[:, i])
        mse_value = mean_squared_error(val[:, i], pred[:, i])
        precision = precision_score(val[:, i], pred[:, i], average='macro')
        recall = recall_score(val[:, i], pred[:, i], average='macro')
        pearson_corr, _ = stats.pearsonr(pred[:, i], val[:, i])
        spearman_corr, _ = stats.spearmanr(pred[:, i], val[:, i])

        print(f"{criteria[i]:^18} | {precision*100:^14.2f} | {recall*100:^12.2f} | {pearson_corr:^10.4f} | {spearman_corr:^10.4f} |")

    global_metrics = {
        'MAE': mean_absolute_error(val.flatten(), pred.flatten()),
        'MSE': mean_squared_error(val.flatten(), pred.flatten()),
        'Precision': precision_score(val.flatten(), pred.flatten(), average='macro'),
        'Recall': recall_score(val.flatten(), pred.flatten(), average='macro'),
        'PLCC': stats.pearsonr(pred.flatten(), val.flatten())[0],
        'SROCC': stats.spearmanr(pred.flatten(), val.flatten())[0]
    }
    
    print(f"\n{'MAE':^10} | {'MSE':^10} | {'Precision (%)':^14} | {'Recall (%)':^12} | {'PLCC':^10} | {'SROCC':^10} |")
    print("-----------------------------------------------------------------------------------")
    print(f"{global_metrics['MAE']:^10.4f} | {global_metrics['MSE']:^10.4f} | {global_metrics['Precision']*100:^14.2f} | {global_metrics['Recall']*100:^12.2f} | {global_metrics['PLCC']:^10.4f} | {global_metrics['SROCC']:^10.4f} |")

def plot_all_confusion_matrices(y_true, y_pred):
    criteria = ['Background', 'Lighting', 'Focus', 'Orientation', 'Color calibration', 'Resolution', 'Field of view']
    fig, axes = plt.subplots(1, 7, figsize=(35, 5), sharey=True)
    for i, ax in enumerate(axes.flatten()):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        ax.title.set_text(criteria[i])
    plt.tight_layout()
    plt.show()