import os
import random
from typing import Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils_data import binarize_scores
from data import GQIDataset


def test(
    root: str = "images",
    crop: bool = True,
    normalize: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    model: torch.nn.Module = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    xgb: Optional[Any] = None,
    regression: bool = False,
):
    seed_it_all()
    dataset = GQIDataset(root=root, crop=crop, normalize=normalize, phase="test")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
    features, scores = get_features_scores(model, dataloader, device, root)
    image_indices = np.arange(len(dataset))
    test_indices = np.repeat(image_indices * 5, 5) + np.tile(np.arange(5), len(image_indices))
    test_features = features[test_indices]
    test_scores = scores[test_indices]
    if regression:
        test_scores = test_scores[::5]  # Scores are repeated for each crop, so we only keep the first one
        predictions = xgb.predict(test_features)
        predictions = np.reshape(predictions, (-1, 5, 7))  # Reshape to group crops per image
        predictions = np.mean(predictions, axis=1) # Average the predictions of the 5 crops of the same image
        predictions = np.clip(predictions, 0, 1)
    
        mae = []
        mse = []
        for i in range(predictions.shape[1]):
            mae_value = mean_absolute_error(test_scores[:, i], predictions[:, i])
            mse_value = mean_squared_error(test_scores[:, i], predictions[:, i])
            mae.append(round(mae_value,4))
            mse.append(round(mse_value,4))
        global_mae_value = mean_absolute_error(test_scores.flatten(), predictions.flatten())
        global_mse_value = mean_squared_error(test_scores.flatten(), predictions.flatten())
    
        correlations = calculate_global_and_metric_correlations(predictions, test_scores)
        print(f"\n{'MAE':<15} {'MSE':<15} {'SROCC':<15} {'PLCC':<15}")
        print(f"{global_mae_value:<15.4f} {global_mse_value:<15.4f} {correlations['global']['spearman']:<15.4f} {correlations['global']['pearson']:<15.4f}\n")
        print("Metric-by-Metric Pearson Correlations:", correlations['by_metric']['pearson'])
        print("Metric-by-Metric Spearman Correlations:", correlations['by_metric']['spearman'], "\n")
    
        plot_prediction_scores(test_scores, predictions, mae, mse)
    else:
        scores = binarize_scores(test_scores)
        predictions = xgb.predict(test_features)
        for i in range(scores.shape[1]):  # Iterate over each class
            print(f"Report for {['Background', 'Lighting', 'Focus', 'Orientation', 'Color calibration', 'Resolution', 'Field of view'][i]}:")
            print(classification_report(scores[:, i], predictions[:, i], labels=[0, 1, 2, 3, 4], target_names=['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4'], zero_division=0))
            
            cm = confusion_matrix(scores[:, i], predictions[:, i], labels=[0, 1, 2, 3, 4])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4'])
            disp.plot()
            plt.title(f"Confusion Matrix for {['Background', 'Lighting', 'Focus', 'Orientation', 'Color calibration', 'Resolution', 'Field of view'][i]}")
            plt.show()
        

def get_features_scores(model: torch.nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        save_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    embed_dir = os.path.join(save_dir, "embeddings")
    feats_file = os.path.join(embed_dir, "features.npy")
    scores_file = os.path.join(embed_dir, "scores.npy")
    
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
            label = batch["label"].repeat(5, 1) # repeat label for each crop
    
            img_orig = rearrange(img_orig, "b n c h w -> (b n) c h w")
            img_ds = rearrange(img_ds, "b n c h w -> (b n) c h w")

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

def calculate_global_and_metric_correlations(predictions, actual_scores):
    if predictions.shape[0] < 2:
        raise ValueError("Not enough samples for correlation calculation. Ensure each metric has at least two samples.")

    pearson_correlations = []
    spearman_correlations = []

    # Calculate correlations for each metric
    for i in range(predictions.shape[1]):
        # Check if there are at least two samples for this metric
        if predictions[:, i].size < 2:
            print(f"Skipping metric {i+1} due to insufficient data.")
            pearson_correlations.append(np.nan)  # Use np.nan or another placeholder to indicate skipped calculation
            spearman_correlations.append(np.nan)
            continue
        
        # Calculate Pearson and Spearman correlations safely
        try:
            pearson_corr, _ = stats.pearsonr(predictions[:, i], actual_scores[:, i])
            spearman_corr, _ = stats.spearmanr(predictions[:, i], actual_scores[:, i])
            pearson_correlations.append(round(pearson_corr,4))
            spearman_correlations.append(round(spearman_corr,4))
        except Exception as e:
            print(f"Error calculating correlations for metric {i+1}: {e}")
            pearson_correlations.append(np.nan)
            spearman_correlations.append(np.nan)

    # Calculate global correlations
    try:
        global_pearson_corr, _ = stats.pearsonr(predictions.flatten(), actual_scores.flatten())
        global_spearman_corr, _ = stats.spearmanr(predictions.flatten(), actual_scores.flatten())
    except Exception as e:
        print(f"Error calculating global correlations: {e}")
        global_pearson_corr, global_spearman_corr = np.nan, np.nan

    return {
        "global": {"pearson": global_pearson_corr, "spearman": global_spearman_corr},
        "by_metric": {"pearson": pearson_correlations, "spearman": spearman_correlations}
    }


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

def seed_it_all(seed=42):
    """ Attempt to be Reproducible """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
