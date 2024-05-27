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
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

from models.multioutput_xgb import MultiOutputXGBClassifier, train_xgbclassifier, sweep_train
from models.multioutput_xgbregressor import MultiOutputXGBRegressor
from utils.visualization import print_metrics, plot_all_confusion_matrices, plot_prediction_scores
from utils.utils_data import binarize_scores, get_features_scores
from data import BaseDataset

def train():
    wandb.init(reinit=True)
    config = wandb.config
    embed_dir = os.path.join(config.root, "embeddings")
    feats_file = os.path.join(embed_dir, f"features_{config.num_distortions}.npy")
    scores_file = os.path.join(embed_dir, f"scores_{config.num_distortions}.npy")
    dataset = BaseDataset(root=config.root, crop=config.crop, phase="train", num_distortions=config.num_distortions)
    
    if os.path.exists(feats_file) and os.path.exists(scores_file):
        features = np.load(feats_file)
        scores = np.load(scores_file)
        print(f'Loaded features from {feats_file}')
        print(f'Loaded scores from {scores_file}')
    else:
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
        arniqa.eval().to(device)
        features, scores = get_features_scores(arniqa, dataloader, device, config.crop)
        if not os.path.exists(embed_dir):
            os.makedirs(embed_dir)
            
        np.save(feats_file, features)
        np.save(scores_file, scores)
        print(f'Saved features to {feats_file}')
        print(f'Saved scores to {scores_file}')
    
    image_indices = np.arange(len(dataset))
    train_img_indices, val_img_indices = train_test_split(image_indices, test_size=0.25, random_state=42, shuffle=True)
    train_features = features[train_img_indices]
    train_scores = scores[train_img_indices]
    val_features = features[val_img_indices]
    val_scores = scores[val_img_indices]

    if config.model_type == 'xgb_reg' or config.model_type == 'xgb_cls':
        params = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'subsample': config.subsample,
            'early_stopping_rounds': config.early_stopping_rounds,
            'gamma': config.gamma,
            'min_child_weight': config.min_child_weight,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'tree_method': 'hist',
            'device': 'cpu',
            'callbacks': [WandbCallback(log_model=True)],
        }
    elif config.model_type == 'mlp_reg' or config.model_type == 'mlp_cls':
        params = {
            'hidden_layer_sizes': config.hidden_layer_sizes,
            'activation': config.activation,
            'solver': 'adam',
            'alpha': config.alpha,
            'learning_rate_init': config.learning_rate_init,
            'max_iter': config.max_iter,
            'early_stopping': True,
        }
    if config.model_type == 'xgb_reg':
        params.update({'objective': "reg:pseudohubererror", 'n_jobs': 16, 'multi_strategy': config.multi_strategy})
        regressor = xgb.XGBRegressor(**params)
        regressor.fit(train_features, train_scores, eval_set=[(val_features, val_scores)], verbose=False)
        
        predictions = regressor.predict(val_features)
        predictions = np.clip(predictions, 0, 1)
        log_srocc(val_scores, predictions)

    elif config.model_type == 'xgb_cls':
        params.update({'booster': 'gbtree', 'objective': 'multi:softprob', 'eval_metric': ['mlogloss', 'merror', 'auc']})
        train_scores = binarize_scores(train_scores)
        val_scores = binarize_scores(val_scores)

        classifier = MultiOutputXGBClassifier(params=params, num_class=5)
        classifier.fit(train_features, train_scores, eval_set=(val_features, val_scores))
        predictions = classifier.predict(val_features)
        log_srocc(val_scores, predictions)
    
    elif config.model_type == 'mlp_reg':
        mlp = MLPRegressor(**params)
        
        multioutput_regressor = MultiOutputRegressor(mlp, n_jobs=-1)
        multioutput_regressor.fit(train_features, train_scores)
        predictions = multioutput_regressor.predict(val_features)
        predictions = np.clip(predictions, 0, 1)
        log_srocc(val_scores, predictions)

    elif config.model_type == 'mlp_cls':
        train_scores = binarize_scores(train_scores)
        val_scores = binarize_scores(val_scores)
        mlp = MLPClassifier(**params)
        
        multioutput_classifier = MultiOutputClassifier(mlp, n_jobs=-1)
        multioutput_classifier.fit(train_features, train_scores)
        predictions = multioutput_classifier.predict(val_features)
        predictions = np.clip(predictions, 0, 1)
        log_srocc(val_scores, predictions)

def log_srocc(val_scores, predictions):
    criteria_labels = ['Background', 'Lighting', 'Focus', 'Orientation', 'Color Calibration', 'Resolution', 'Field of View']
    spearman_scores = []
    for i, label in enumerate(criteria_labels):
        spearman_corr = stats.spearmanr(val_scores[:, i], predictions[:, i]).correlation
        wandb.log({f'srocc_{label}': spearman_corr})
        spearman_scores.append(spearman_corr)

    overall_srocc = stats.spearmanr(predictions.flatten(), val_scores.flatten()).correlation
    wandb.log({"overall_srocc": overall_srocc})

    return spearman_scores, overall_srocc