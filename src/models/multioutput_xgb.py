import xgboost as xgb
import numpy as np
import wandb
from wandb.integration.xgboost import WandbCallback
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats

class MultiOutputXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params, num_class):
        self.models = []
        self.params = params
        self.num_class = num_class

    def fit(self, X, y, eval_set=None):
        criteria_labels = ['Background', 'Lighting', 'Focus', 'Orientation', 'Color Calibration', 'Resolution', 'Field of View']
        spearman_scores = []
        for i in range(y.shape[1]):
            #print(f"Training for label {i+1}")
            print(f"Training for criteria: {criteria_labels[i]}")
            model = xgb.XGBClassifier(**self.params)
            if eval_set is not None:
                model.fit(X, y[:, i], eval_set=[(eval_set[0], eval_set[1][:, i])], verbose=False)
            else:
                model.fit(X, y[:, i])
            self.models.append(model)

            if wandb.run:
                predictions = model.predict(eval_set[0])
                spearman_corr = stats.spearmanr(eval_set[1][:, i], predictions).correlation
                wandb.log({f'srocc_{criteria_labels[i]}': spearman_corr})
                spearman_scores.append(spearman_corr)
        if wandb.run:            
            mean_spearman = np.mean(spearman_scores)
            wandb.log({'mean_spearman': mean_spearman})
        
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return predictions

    def predict_proba(self, X):
        proba_predictions = np.column_stack([model.predict_proba(X) for model in self.models])
        return proba_predictions

def train_xgbclassifier(train_features, train_scores, val_features, val_scores, params, use_wandb=False, use_sweep=False):
    if use_wandb:
        if use_sweep:
            sweep_config = {
                "method": "grid",
                "metric": {
                    "name": "mean_spearman",
                    "goal": "maximize"
                },
                "parameters": {
                    "n_estimators": {"values": [50, 100, 200, 300]},
                    "max_depth": {"values": [3, 6, 9, 12]},
                    "learning_rate": {"values": [0.3, 0.1, 0.05, 0.005]},
                    "subsample": {"values": [1, 0.7, 0.5, 0.3]}
                }
            }
            sweep_id = wandb.sweep(sweep_config, project="BAA", entity="choekyel-hslu")
            # params.update({'callbacks': [WandbCallback(log_model=True)]})
            wandb.agent(sweep_id, lambda: sweep_train(train_features, train_scores, val_features, val_scores, params), count=1)
            classifier, predictions, val_scores = train(train_features, train_scores, val_features, val_scores, params)
        else:
            wandb.init(project="BAA", entity="choekyel-hslu", config=params)
            classifier, predictions, val_scores = train(train_features, train_scores, val_features, val_scores, params)
            wandb.finish()
        return classifier, predictions, val_scores
    else:
        return train(train_features, train_scores, val_features, val_scores, params)

def sweep_train(train_features, train_scores, val_features, val_scores, params):
    wandb.init(config=params, reinit=True)
    classifier, predictions, val_scores = train(train_features, train_scores, val_features, val_scores, params)
    
    
def train(train_features, train_scores, val_features, val_scores, params):
    classifier = MultiOutputXGBClassifier(params=params, num_class=5)
    classifier.fit(train_features, train_scores, eval_set=(val_features, val_scores))
    predictions = classifier.predict(val_features)
    if wandb.run:
        overall_srocc = stats.spearmanr(predictions.flatten(), val_scores.flatten())[0]
        criteria_sroccs = [stats.spearmanr(predictions[:, i], val_scores[:, i])[0] for i in range(predictions.shape[1])]
        wandb.log({"overall_srocc": overall_srocc, "criteria_sroccs": criteria_sroccs})
    return classifier, predictions, val_scores