# train.py
# Run: python train.py --config_path config.yaml

import os
import random
import argparse
import yaml
import numpy as np
import torch
import wandb
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from models.multioutput_xgb import train_xgbclassifier
from utils.visualization import print_metrics, plot_all_confusion_matrices, plot_prediction_scores
from utils.utils_data import discretization, get_features_scores
from data import BaseDataset

def train_model(config):
    seed_it_all()
    embed_dir = os.path.join(config['root'], "embeddings")
    feats_file = os.path.join(embed_dir, f"features_{config['num_distortions']}.npy")
    scores_file = os.path.join(embed_dir, f"scores_{config['num_distortions']}.npy")
    
    dataset = BaseDataset(root=config['root'], phase="train", num_distortions=config['num_distortions'])
    if os.path.exists(feats_file) and os.path.exists(scores_file):
        features = np.load(feats_file)
        scores = np.load(scores_file)
        print(f'Loaded features from {feats_file}')
        print(f'Loaded scores from {scores_file}')
    else:
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        arniqa = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
        arniqa.eval().to(device)
        features, scores = get_features_scores(arniqa, dataloader, device)
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
        
    if config['model_type'] == 'xgb_reg':
        params = {
            'tree_method': "hist",
            'n_estimators': config.get('n_estimators', 100),
            'n_jobs': 16,
            'max_depth': config.get('max_depth', 6),
            'min_child_weight': config.get('min_child_weight', 1),
            'learning_rate': config.get('learning_rate', 0.01),
            'subsample': config.get('subsample', 1.0),
            'early_stopping_rounds': config.get('early_stopping_rounds', 10),
            'reg_alpha': 0,
            'reg_lambda': 1,
            'gamma': config.get('gamma', 0),
            'objective': "reg:pseudohubererror",
            'multi_strategy': config.get('multi_strategy', "one_output_per_tree"),
        }
        regressor = xgb.XGBRegressor(**params)
        regressor.fit(train_features, train_scores, eval_set=[(val_features, val_scores)], verbose=False)
        predictions = regressor.predict(val_features)
        predictions = np.clip(predictions, 0, 1)
        bin_pred = discretization(predictions)
        bin_val = discretization(val_scores)
        plot_prediction_scores(val_scores, predictions)
        plot_all_confusion_matrices(bin_val, bin_pred)
        print_metrics(bin_val, bin_pred)
        return regressor

    elif config['model_type'] == 'xgb_cls':
        train_scores = discretization(train_scores)
        val_scores = discretization(val_scores)
        params = {
            'booster': 'gbtree',
            'n_estimators': config.get('n_estimators', 300),
            'max_depth': config.get('max_depth', 6),
            'min_child_weight': config.get('min_child_weight', 1),
            'learning_rate': config.get('learning_rate', 0.01),
            'subsample': config.get('subsample', 1.0),
            'objective': 'multi:softprob',
            'random_state': 42,
            'eval_metric': ['mlogloss', 'merror', 'auc'],
            'early_stopping_rounds': config.get('early_stopping_rounds', 10),
            'reg_alpha': 0,
            'reg_lambda': 1,
            'gamma': config.get('gamma', 0),
            'tree_method': 'hist',
            'device': 'cpu'
        }
        classifier, predictions, val_scores = train_xgbclassifier(train_features, train_scores, val_features, val_scores, params, use_wandb=config['logging']['use_wandb'], use_sweep=False)
        plot_all_confusion_matrices(val_scores, predictions)
        plot_prediction_scores(val_scores, predictions)
        print_metrics(val_scores, predictions)
        return classifier
        
    elif config['model_type'] == 'mlp_reg':
        params = {
            'hidden_layer_sizes': tuple(config.get('hidden_layer_sizes', [100])),
            'activation': config.get('activation', 'relu'),
            'solver': 'adam',
            'alpha': config.get('alpha', 0.0001),
            'learning_rate_init': config.get('learning_rate_init', 0.001),
            'max_iter': config.get('max_iter', 200),
            'early_stopping': True,
        }
        mlp = MLPRegressor(**params)
        multioutput_regressor = MultiOutputRegressor(mlp, n_jobs=-1)
        multioutput_regressor.fit(train_features, train_scores)
        predictions = multioutput_regressor.predict(val_features)
        predictions = np.clip(predictions, 0, 1)
        bin_pred = discretization(predictions)
        bin_val = discretization(val_scores)
        plot_prediction_scores(val_scores, predictions)
        plot_all_confusion_matrices(bin_val, bin_pred)
        print_metrics(bin_val, bin_pred)
        return multioutput_regressor
    
    elif config['model_type'] == 'mlp_cls':
        params = {
            'hidden_layer_sizes': tuple(config.get('hidden_layer_sizes', [100])),
            'activation': config.get('activation', 'relu'),
            'solver': 'adam',
            'alpha': config.get('alpha', 0.0001),
            'learning_rate_init': config.get('learning_rate_init', 0.001),
            'max_iter': config.get('max_iter', 200),
            'early_stopping': True,
        }
        train_scores = discretization(train_scores)
        val_scores = discretization(val_scores)
        mlp = MLPClassifier(**params)
        multioutput_classifier = MultiOutputClassifier(mlp, n_jobs=-1)
        multioutput_classifier.fit(train_features, train_scores)
        predictions = multioutput_classifier.predict(val_features)
        plot_all_confusion_matrices(val_scores, predictions)
        plot_prediction_scores(val_scores, predictions)
        print_metrics(val_scores, predictions)
        return multioutput_classifier
    

def seed_it_all(seed=42):
    """ Attempt to be Reproducible """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sweep_train():
    wandb.init(reinit=True)
    config = wandb.config
    train_model(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for image quality assessment")
    parser.add_argument('–config_path', type=str, required=True, help='Path to the config.yaml file')
                        
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)['train']

    if config['sweep']:
        sweep_id = wandb.sweep(config['sweep_config'], project=config['logging']['wandb']['project'], entity=config['logging']['wandb']['entity'])
        wandb.agent(sweep_id, sweep_train, count=config['sweep_count'])
        wandb.finish()
    else:
        if config['logging']['use_wandb']:
            wandb.init(project=config['logging']['wandb']['project'], entity=config['logging']['wandb']['entity'])
            train_model(config)
            wandb.finish()
        else:
            train_model(config)