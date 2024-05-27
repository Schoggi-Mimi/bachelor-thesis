import os
import random
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
import wandb
from wandb.integration.xgboost import WandbCallback
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
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
from models.multioutput_xgbregressor import MultiOutputXGBRegressor
from utils.utils_data import map_predictions_to_intervals, binarize_scores, get_features_scores
from utils.visualization import print_metrics, plot_all_confusion_matrices, plot_prediction_scores
from data import BaseDataset

def train_model(
    root: str = "SCIN",
    num_distortions: int = 10,
    crop: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    model_type: str = 'reg',
    track: bool = False,
):
    seed_it_all()
    assert model_type in ["xgb_reg", "xgb_cls", "mlp_reg", "mlp_cls"], "phase must be in ['xgb_reg', 'xgb_cls', 'mlp_reg', 'mlp_cls']"

    embed_dir = os.path.join(root, "embeddings")
    feats_file = os.path.join(embed_dir, f"features_{num_distortions}.npy")
    scores_file = os.path.join(embed_dir, f"scores_{num_distortions}.npy")
    
    dataset = BaseDataset(root=root, crop=crop, phase="train", num_distortions=num_distortions)
    if os.path.exists(feats_file) and os.path.exists(scores_file):
        features = np.load(feats_file)
        scores = np.load(scores_file)
        print(f'Loaded features from {feats_file}')
        print(f'Loaded scores from {scores_file}')
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
        arniqa.eval().to(device)
        features, scores = get_features_scores(arniqa, dataloader, device, crop)
        if not os.path.exists(embed_dir):
            os.makedirs(embed_dir)
            
        np.save(feats_file, features)
        np.save(scores_file, scores)
        print(f'Saved features to {feats_file}')
        print(f'Saved scores to {scores_file}')

    image_indices = np.arange(len(dataset))
    train_img_indices, val_img_indices = train_test_split(image_indices, test_size=0.25, random_state=42, shuffle=True)
    if crop:
        train_img_indices = np.repeat(train_img_indices * 5, 5) + np.tile(np.arange(5), len(train_img_indices))
        val_img_indices = np.repeat(val_img_indices * 5, 5) + np.tile(np.arange(5), len(val_img_indices))

    train_features = features[train_img_indices]
    train_scores = scores[train_img_indices]
    val_features = features[val_img_indices]
    val_scores = scores[val_img_indices]
    if crop:
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one
        
    if model_type == 'xgb_reg':
        params = { # num_distortions = 16
            'tree_method': "hist",
            'n_estimators': 50,
            'n_jobs': 16,
            'max_depth': 7,
            'min_child_weight': 81,
            'learning_rate': 0.06891,
            'subsample': 0.9,
            'early_stopping_rounds': 30,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'gamma': 0.1289,
            'objective': "reg:pseudohubererror",
            'multi_strategy': "one_output_per_tree", # one_output_per_tree, multi_output_tree
        }
        params0 = { # Gaussian, 30
            'tree_method': "hist",
            'n_estimators': 300,
            'n_jobs': 16,
            'max_depth': 9,
            'min_child_weight': 149,
            'learning_rate': 0.07046,
            'subsample': 0.8,
            'early_stopping_rounds': 30,
            'reg_alpha': 0,
            'reg_lambda': 1,
            "gamma": 0.2615,
            'objective': "reg:pseudohubererror",
        }
        #regressor = MultiOutputXGBRegressor(params=params)
        #regressor.fit(train_features, train_scores, eval_set=(val_features, val_scores))
        regressor = xgb.XGBRegressor(**params)
        #regressor = MultiOutputRegressor(regressor)
        regressor.fit(train_features, train_scores, eval_set=[(val_features, val_scores)], verbose=False)
        predictions = regressor.predict(val_features)
        predictions = np.clip(predictions, 0, 1)
        bin_pred = binarize_scores(predictions)
        bin_val = binarize_scores(val_scores)
        plot_prediction_scores(val_scores, predictions)
        plot_all_confusion_matrices(bin_val, bin_pred)
        print_metrics(bin_val, bin_pred)
        return regressor

    elif model_type == 'xgb_cls':
        train_scores = binarize_scores(train_scores)
        val_scores = binarize_scores(val_scores)
        params = {
            'booster': 'gbtree',
            'n_estimators': 300,
            'max_depth': 9,
            'min_child_weight': 44,
            'learning_rate': 0.2775,
            'subsample': 0.8,
            'objective': 'multi:softprob',
            'random_state': 42,
            'eval_metric': ['mlogloss', 'merror', 'auc'],
            'early_stopping_rounds': 30,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'gamma': 0.2738,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        classifier, predictions, val_scores = train_xgbclassifier(train_features, train_scores, val_features, val_scores, params, use_wandb=track, use_sweep=False)
        plot_all_confusion_matrices(val_scores, predictions)
        plot_prediction_scores(val_scores, predictions)
        print_metrics(val_scores, predictions)
        return classifier
        
    elif model_type == 'mlp_reg':
        params1 = { # num_distortion = 32
            'hidden_layer_sizes': (1024,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.008267,
            'learning_rate_init': 0.09256,
            'max_iter': 500,
            'early_stopping': True,
        }
        params1 = { # num_distortion = 64
            'hidden_layer_sizes': (512,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001446,
            'learning_rate_init': 0.01646,
            'max_iter': 300,
            'early_stopping': True,
        }
        params = { # num_distortion = 64
            'hidden_layer_sizes': (512,256),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.00234,
            'learning_rate_init': 0.002089,
            'max_iter': 500,
            'early_stopping': True,
        }
        mlp = MLPRegressor(**params)
        multioutput_regressor = MultiOutputRegressor(mlp, n_jobs=-1)
        multioutput_regressor.fit(train_features, train_scores)
        predictions = multioutput_regressor.predict(val_features)
        predictions = np.clip(predictions, 0, 1)
        bin_pred = binarize_scores(predictions)
        bin_val = binarize_scores(val_scores)
        plot_prediction_scores(val_scores, predictions)
        plot_all_confusion_matrices(bin_val, bin_pred)
        print_metrics(bin_val, bin_pred)
        return multioutput_regressor

    elif model_type == 'mlp_cls':
        params = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.008267,
            'learning_rate_init': 0.09256,
            'max_iter': 500,
            'early_stopping': True,
        }
        train_scores = binarize_scores(train_scores)
        val_scores = binarize_scores(val_scores)
        mlp = MLPClassifier(**params)
        multioutput_classifier = MultiOutputClassifier(mlp, n_jobs=-1)
        multioutput_classifier.fit(train_features, train_scores)
        predictions = multioutput_classifier.predict(val_features)
        plot_all_confusion_matrices(val_scores, predictions)
        plot_prediction_scores(val_scores, predictions)
        print_metrics(val_scores, predictions)
        return multioutput_regressor
        
        
def seed_it_all(seed=42):
    """ Attempt to be Reproducible """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
