from datetime import datetime, timezone, timedelta
import os
import time
import pandas as pd
import numpy as np
import logging
import sklearn.calibration
import csv

from collections import deque

from lightgbm import LGBMClassifier

from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)



from poly_market_maker.dataset import Dataset
from poly_market_maker.models import Model
from poly_market_maker.tensorflow_classifier import TensorflowClassifier
from poly_market_maker.bucket_classifier import BucketClassifier

SPREAD = 0.05
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def evaluate_model(model, test_df, show_results=True):
    print(f"Evaluating model: {model.name}")
    feature_cols = model.feature_cols
    probs = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df['probability'] = np.clip(probs, 0.0, 1.0)
    y_true = test_df['label'].astype(float)
    y_pred_proba = test_df['probability'].astype(float)
    test_df['y_true'] = y_true
    test_df['y_pred_proba'] = y_pred_proba

    # Bucket price_delta into 10 bins
    test_df['delta_bucket'] = pd.qcut(test_df['delta'], 20, duplicates='drop')
    
    # Bucket seconds_left into 5 bins
    test_df['seconds_left_bucket'] = pd.qcut(test_df['seconds_left'], 15, duplicates='drop')

    print(f"Test dataframe: {test_df.head()}")

    # Calculate PnL for each group
    group_results = []
    for (d_bucket, t_bucket), group in test_df.groupby(['delta_bucket', 'seconds_left_bucket'], observed=False):
        print(f"Group: {d_bucket}, {t_bucket}")
        action = (group['probability'] - SPREAD > group['bid']) & (group['bid'] > 0)
        buy_trades = group[action].copy()
        
        buy_trades['revenue'] = buy_trades['is_up'].astype(float)
        buy_trades['cost'] = buy_trades['bid'].astype(float)
        buy_trades['pnl'] = buy_trades['revenue'] - buy_trades['cost']
        
        group_results.append({
            'seconds_left_bucket': t_bucket,
            'delta_bucket': d_bucket,
            'opportunity': group['is_up'].sum() - group['bid'].sum(),
            
            'prob_est': round(group['prob_est'].mean(), 3),
            'average_is_up': round(group['is_up'].mean(), 3),
            'average_probability': round(group['probability'].mean(), 3),
            'average_bid': round(group['bid'].mean(), 3),

            'pnl': buy_trades['pnl'].sum(),  
            'buy_trades': len(buy_trades),
            'num_rows': len(group),
        })
    

    
    # Create results dataframe and sort by increasing PnL
    results_df = pd.DataFrame(group_results)
    # results_df = results_df.sort_values(['seconds_left_bucket', 'delta_bucket'], ascending=[False, True])
    results_df = results_df.sort_values(['seconds_left_bucket', 'opportunity'], ascending=[False, True])
    
    if show_results:
        print("\nGroups sorted by seconds_left_bucket and delta_bucket:")
        print(results_df.to_string(index=False))

    ret = model.dataset.evaluate_model_metrics(test_df, probability_column='probability', threshold=0.5, spread=SPREAD)
    ret['model'] = model.name
    print(ret)
    return ret

def calibrate_model(rf_model):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, brier_score_loss

    model = Pipeline([
        ("quant", QuantileTransformer(output_distribution="normal")),
        ("rf", rf_model),
    ])

    calib_model = CalibratedClassifierCV(model, method="sigmoid", cv=5)
    return calib_model

def main():
    dataset = Dataset()
    
    train_df = dataset.train_df
    test_df = dataset.test_df

    feature_cols = ['delta', 'seconds_left', 'prob_est']
    # feature_cols = ['log_return', 'time']

    # random_forest_params = {'n_estimators': 253, 'max_depth': 40, 'min_samples_split': 54, 'min_samples_leaf': 33, 'max_features': 0.5, 'bootstrap': False, 'class_weight': 'balanced'}
    random_forest_params = {'n_estimators': 1674, 'max_depth': 35, 'min_samples_split': 86, 'min_samples_leaf': 177, 'max_features': 'log2', 'bootstrap': False, 'class_weight': 'balanced'}
    random_forest_params = {'n_estimators': 1448, 'max_depth': 36, 'min_samples_split': 130, 'min_samples_leaf': 135, 'max_features': 'log2', 'bootstrap': False, 'class_weight': 'balanced_subsample'}
    random_forest_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 50, 'min_samples_leaf': 50, 'max_features': 'log2', 'bootstrap': False, 'class_weight': 'balanced_subsample'}

    # model = Model(f"RandomForestClassifier_features_{len(feature_cols)}_{random_forest_params['n_estimators']}_{random_forest_params['max_depth']}_{random_forest_params['min_samples_split']}_{random_forest_params['min_samples_leaf']}_{random_forest_params['max_features']}_{random_forest_params['bootstrap']}_{random_forest_params['class_weight']}", RandomForestClassifier(**random_forest_params), feature_cols=feature_cols, dataset=dataset)
    # model.load()
    # print(model.model)
    models = [
        # Model("CalibratedRandomForestClassifier", calibrate_model(model.model), feature_cols=feature_cols, dataset=dataset),
        # Model(f"RandomForestClassifier_900_32_7", RandomForestClassifier(**random_forest_params), feature_cols=feature_cols, dataset=dataset),
        Model(f"RandomForestClassifier_features_{len(feature_cols)}_{random_forest_params['n_estimators']}_{random_forest_params['max_depth']}_{random_forest_params['min_samples_split']}_{random_forest_params['min_samples_leaf']}_{random_forest_params['max_features']}_{random_forest_params['bootstrap']}_{random_forest_params['class_weight']}", RandomForestClassifier(**random_forest_params), feature_cols=feature_cols, dataset=dataset),
        
        # Model("LogisticRegression", LogisticRegression(max_iter=300, C=0.5, solver='lbfgs'), feature_cols=feature_cols, dataset=dataset),
        # Model("LGBMClassifier", LGBMClassifier(n_estimators=300, max_depth=-1, learning_rate=0.001, random_state=42,), feature_cols=feature_cols, dataset=dataset),
        # Model("GradientBoostingClassifier_features_{len(feature_cols)}", GradientBoostingClassifier(n_estimators=300, learning_rate=0.001, max_depth=3, random_state=42), feature_cols=feature_cols, dataset=dataset),
        # Model("BucketClassifier", BucketClassifier(), feature_cols=feature_cols, dataset=dataset),
    ]


    for model in models:
        # evaluate_model(model, train_df, show_results=True)
        evaluate_model(model, dataset.df, show_results=True)

if __name__ == "__main__":
    main()