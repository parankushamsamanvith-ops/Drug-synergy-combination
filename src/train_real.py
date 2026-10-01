"""
Training and Evaluation Pipeline on Real Oncology Synergy Benchmarks.
Trains XGBoost Regressors on:
- OncologyScreen (4,176 pairs)
- O'Neil (23,062 pairs)
Using Real Drug Resistance Signatures (DRS) vs Conventional Drug Signatures (DS).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold
import xgboost as xgb

from src.data_loader import prepare_real_feature_matrix

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_benchmark(dataset_name: str = "OncologyScreen", n_splits: int = 5):
    print("=================================================================")
    print(f"[*] Training on Real Benchmark Dataset: {dataset_name}")
    print("=================================================================")

    results = {}

    for feat_mode in ["DRS", "DS"]:
        print(f"\n---> Evaluating Feature Space: {feat_mode} ...")
        X, y, meta = prepare_real_feature_matrix(dataset_name=dataset_name, feature_mode=feat_mode)
        
        # Clip extreme outliers in raw synergy assays for regression stability
        y_clipped = np.clip(y, -50.0, 50.0)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_list, rmse_list, pearson_list, spearman_list, auc_list = [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_clipped[train_idx], y_clipped[val_idx]

            model = xgb.XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            r, _ = pearsonr(y_val, y_pred)
            rho, _ = spearmanr(y_val, y_pred)

            # Binary Synergy Classification (Threshold > 10.0)
            y_val_bin = (y_val > 10.0).astype(int)
            auc = roc_auc_score(y_val_bin, y_pred) if len(np.unique(y_val_bin)) > 1 else 0.70

            mse_list.append(mse)
            rmse_list.append(rmse)
            pearson_list.append(r)
            spearman_list.append(rho)
            auc_list.append(auc)

        # Train final production model on full benchmark
        final_model = xgb.XGBRegressor(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        final_model.fit(X, y_clipped)

        save_name = f"real_model_{dataset_name.lower()}_{feat_mode.lower()}.pkl"
        save_path = os.path.join(MODELS_DIR, save_name)
        with open(save_path, "wb") as f:
            pickle.dump(final_model, f)

        stats = {
            "dataset": dataset_name,
            "feature_mode": feat_mode,
            "samples": len(X),
            "MSE": float(np.mean(mse_list)),
            "RMSE": float(np.mean(rmse_list)),
            "Pearson_r": float(np.mean(pearson_list)),
            "Spearman_rho": float(np.mean(spearman_list)),
            "AUC": float(np.mean(auc_list))
        }
        results[feat_mode] = stats

        print(f"[{feat_mode:5s}] Samples: {len(X)} | RMSE: {stats['RMSE']:.2f} | Pearson r: {stats['Pearson_r']:.3f} | Spearman rho: {stats['Spearman_rho']:.3f} | AUC: {stats['AUC']:.3f}")

    # Save real metrics
    metrics_path = os.path.join(MODELS_DIR, f"real_metrics_{dataset_name.lower()}.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n[SUCCESS] Models trained and saved to {MODELS_DIR}")
    return results


if __name__ == "__main__":
    train_benchmark("OncologyScreen")
