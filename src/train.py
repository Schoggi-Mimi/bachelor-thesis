import os
import random
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from xgboost import XGBRegressor, XGBClassifier
import wandb
from wandb.integration.xgboost import WandbCallback
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from models.multioutput_xgb import MultiOutputXGBClassifier, train_xgbclassifier
from utils.utils_data import map_predictions_to_intervals, binarize_scores
from data import GQIDataset, BaseDataset

def train_model(
    root: str = "images",
    num_distortions: int = 10,
    crop: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    model: torch.nn.Module = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    regression: bool = False,
    track: bool = False,
):
    seed_it_all()

    # dataset = GQIDataset(root=root, crop=crop, phase="train")
    dataset = BaseDataset(root=root, crop=crop, phase="train", num_distortions=num_distortions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    features, scores = get_features_scores(model, dataloader, device, root, crop, num_distortions)

    image_indices = np.arange(len(dataset))
    train_img_indices, val_img_indices = train_test_split(image_indices, test_size=0.25, random_state=42, shuffle=True)
    if crop:
        train_img_indices = np.repeat(train_img_indices * 5, 5) + np.tile(np.arange(5), len(train_img_indices))
        val_img_indices = np.repeat(val_img_indices * 5, 5) + np.tile(np.arange(5), len(val_img_indices))

    train_features = features[train_img_indices]
    train_scores = scores[train_img_indices]
    val_features = features[val_img_indices]
    val_scores = scores[val_img_indices]
    
    if regression:
        params = {
            "n_estimators": 500, # [1, inf)
            "max_depth": 4, # [1, inf)
            "min_samples_split": 10, # [2, inf)
            "learning_rate": 0.01,
            "loss": "squared_error",
            "random_state": 42,
            "verbose": 1,
        }
        #regressor = MultiOutputRegressor(GradientBoostingRegressor(**params)).fit(train_features, train_scores)
        regressor = XGBRegressor(
            tree_method="hist",
            n_estimators=128,
            n_jobs=16,
            max_depth=6,# 8
            learning_rate=0.01,
            multi_strategy="multi_output_tree",
            subsample=0.9, # prevent overfitting
            early_stopping_rounds=10,
        ).fit(train_features, train_scores, eval_set=[(train_features, train_scores), (val_features, val_scores)], verbose=True)
        #regressor = MLPRegressor(hidden_layer_sizes=(2048, 1024), activation='relu', solver='adam', max_iter=200, random_state=42, verbose=True).fit(train_features, train_scores) # (2048, 1024, 512), (1024, 512), (512, 256), (2048, 1024, 512, 256, 128, 64)
        if crop:
            val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one
        predictions = regressor.predict(val_features)
        predictions = np.reshape(predictions, (-1, 5, 7))  # Reshape to group crops per image
        predictions = np.mean(predictions, axis=1) # Average the predictions of the 5 crops of the same image
        predictions = np.clip(predictions, 0, 1)
        
        predictions = map_predictions_to_intervals(predictions)
        bin_pred = binarize_scores(predictions)
        bin_val = binarize_scores(val_scores)
        print_metrics(bin_val, bin_pred)

        return regressor

    else:
        train_scores = binarize_scores(train_scores)
        val_scores = binarize_scores(val_scores)
        params = {
            'booster': 'gbtree',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.6,
            'objective': 'multi:softprob',
            'random_state': 42,
            'eval_metric': ['mlogloss', 'merror', 'auc'],
            'early_stopping_rounds': 10,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'tree_method': 'hist',
            'device': 'cuda'
        }
        classifier, predictions, val_scores = train_xgbclassifier(train_features, train_scores, val_features, val_scores, params, use_wandb=track, use_sweep=False)
        plot_all_confusion_matrices(val_scores, predictions)
        print_metrics(val_scores, predictions)
        return classifier, predictions, val_scores

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

def plot_prediction_scores(val_scores, predictions, mae, mse):
    """
    Plot actual vs. predicted scores with annotations for Mean Absolute Error (MAE) and Mean Squared Error (MSE).

    Parameters:
        val_scores (np.array): The actual validation scores, expected shape (n_samples, n_metrics).
        predictions (np.array): The predicted scores, expected shape (n_samples, n_metrics).
        mae (list): List of Mean Absolute Error values for each score metric.
        mse (list): List of Mean Squared Error values for each score metric.
        distortion_criteria (list): List of labels for the score metrics.

    Returns:
        None
    """
    distortion_criteria = ["Background", "Lighting", "Focus", "Orientation", "Color calibration", "Resolution", "Field of view"]
    fig, axes = plt.subplots(1, len(distortion_criteria), figsize=(20, 4), sharey=True, sharex=True)
    for i, ax in enumerate(axes):
        # Scatter plot of actual vs. predicted scores
        ax.scatter(val_scores[:, i], predictions[:, i], color='blue', alpha=0.5, edgecolors='none')
        ax.plot([val_scores[:, i].min(), val_scores[:, i].max()], [val_scores[:, i].min(), val_scores[:, i].max()], 'r--', lw=2)
        ax.set_title(f'Score Metric {i+1} ({distortion_criteria[i]})')
        ax.set_xlabel('Actual Scores')
        ax.set_ylabel('Predicted Scores')
        
        # Add text annotations for MAE and MSE
        textstr = f'MAE: {mae[i]:.4f}\nMSE: {mse[i]:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()

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
    
def seed_it_all(seed=42):
    """ Attempt to be Reproducible """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
