import os
import random
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from data import GQIDataset

def train_regressor(
    root: str = "images",
    crop: bool = True,
    normalize: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    model: torch.nn.Module = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir: str = 'features',
    cross_val: bool = False,
):
    seed_it_all()

    dataset = GQIDataset(root=root, crop=crop, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    features, scores = get_features_scores(model, dataloader, device, save_dir)

    image_indices = np.arange(len(dataset))
    train_img_indices, val_img_indices = train_test_split(image_indices, test_size=0.25, random_state=42, shuffle=True)
    train_indices = np.repeat(train_img_indices * 5, 5) + np.tile(np.arange(5), len(train_img_indices))
    val_indices = np.repeat(val_img_indices * 5, 5) + np.tile(np.arange(5), len(val_img_indices))

    train_features = features[train_indices]
    train_scores = scores[train_indices]
    val_features = features[val_indices]
    val_scores = scores[val_indices]
    val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one
    orig_val_indices = val_indices[::5] // 5  # Get original indices

    #scaler = StandardScaler()
    #train_features_scaled = scaler.fit_transform(train_features)
    #val_features_scaled = scaler.transform(val_features)
    
    #regressor = LinearRegression().fit(train_features, train_scores)
    #regressor = RandomForestRegressor(n_estimators=50, random_state=42).fit(train_features, train_scores)
    #regressor = MultiOutputRegressor(GradientBoostingRegressor(loss='huber', n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42, verbose=1)).fit(train_features, train_scores)
    #regressor = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(train_features, train_scores)
    #regressor = MultiOutputRegressor(SVR(kernel='rbf', C=50, gamma='auto', verbose=True)).fit(train_features, train_scores)
    #regressor = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate='adaptive', max_iter=200, random_state=42, verbose=True).fit(train_features, train_scores)
    regressor = MLPRegressor(hidden_layer_sizes=(2048, 1024, 512), activation='relu', solver='adam', learning_rate='adaptive', max_iter=200, random_state=42, verbose=True).fit(train_features, train_scores) # (2048, 1024, 512), (1024, 512), (512, 256)
    

    if cross_val:
        predictions = cross_val_score(regressor, val_features, val_scores, cv=5, scoring='neg_mean_squared_error')
        predictions = -predictions
        print("Cross-validation MSE scores:", predictions)
        print("Average CV MSE:", np.mean(predictions))

    predictions = regressor.predict(val_features)
    predictions = np.reshape(predictions, (-1, 5, 7))  # Reshape to group crops per image
    predictions = np.mean(predictions, axis=1) # Average the predictions of the 5 crops of the same image
    predictions = np.clip(predictions, 0, 1)

    mae = []
    mse = []
    
    for i in range(predictions.shape[1]):
        mae_value = mean_absolute_error(val_scores[:, i], predictions[:, i])
        mse_value = mean_squared_error(val_scores[:, i], predictions[:, i])
    
        mae.append(mae_value)
        mse.append(mse_value)
        
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)

    correlations = calculate_global_and_metric_correlations(predictions, val_scores)
    print("Global Pearson Correlation:", correlations['global']['pearson'])
    print("Global Spearman Correlation:", correlations['global']['spearman'])
    print("Metric-by-Metric Pearson Correlations:", correlations['by_metric']['pearson'])
    print("Metric-by-Metric Spearman Correlations:", correlations['by_metric']['spearman'])

    plot_prediction_scores(val_scores, predictions, mae, mse)

    return regressor, val_scores, predictions

def get_features_scores(model: torch.nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        save_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    feats_file = os.path.join(save_dir, "features.npy")
    scores_file = os.path.join(save_dir, "scores.npy")
    
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
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    np.save(feats_file, feats)
    np.save(scores_file, scores)
    print(f'Saved features to {feats_file}')
    print(f'Saved scores to {scores_file}')
    
    return feats, scores

import numpy as np
from scipy import stats
import torch
from sklearn.model_selection import train_test_split

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
            pearson_correlations.append(pearson_corr)
            spearman_correlations.append(spearman_corr)
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
    distortion_criteria = ["Lighting", "Focus", "Orientation", "Color calibration", "Background", "Resolution", "Field of view"]
    fig, axes = plt.subplots(1, len(distortion_criteria), figsize=(20, 4), sharey=True)
    for i, ax in enumerate(axes):
        ax.scatter(range(len(val_scores[:, i])), val_scores[:, i], color='blue', label='Actual')
        ax.scatter(range(len(predictions[:, i])), predictions[:, i], color='red', label='Prediction', alpha=0.6)
        ax.set_title(f'Score Metric {i+1} ({distortion_criteria[i]})')
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Scores')
        ax.legend(loc='lower right')

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
