"""
train.py
=========
Trains a stacking ensemble (XGBoost + LightGBM + RandomForest)
on the pre-processed NIFTY 50 data.

Key design decisions
────────────────────
• Target  : next-day Daily_Return  (shift(-1) on scaled column)
• Split   : 80/20 chronological (no shuffling – preserves time order)
• CV      : TimeSeriesSplit(5) used only for OOF meta-features
• Output  : sklearn-compatible joblib pickle (.pkl)
            → evaluate.py / deploy.py / model_loader.py all use joblib.load()
"""

from __future__ import annotations

import json
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("⚠️  LightGBM not installed – ensemble will use XGB + RF + ET only.")

from preprocess import FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Thin sklearn-compatible wrapper so joblib.load() → .predict() works
# ─────────────────────────────────────────────────────────────────────────────
class WeightedEnsemble:
    """Weighted average of N sklearn-compatible regressors."""

    def __init__(self, models: list, weights: list | None = None):
        self.models  = models
        self.weights = weights or [1.0 / len(models)] * len(models)

    def predict(self, X) -> np.ndarray:
        preds = np.column_stack([m.predict(X) for m in self.models])
        w     = np.array(self.weights)
        return preds @ w


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-fold stacking helper
# ─────────────────────────────────────────────────────────────────────────────
def _oof_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    """Returns out-of-fold predictions for the training set."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof  = np.zeros(len(X))
    for train_idx, val_idx in tscv.split(X):
        clone = joblib.loads(joblib.dumps(model))   # deep copy
        clone.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clone.predict(X[val_idx])
    return oof


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    input_path:        str,
    model_output_path: str,
    metrics_path:      str | None = None,
) -> float:
    """
    Trains the full stacking ensemble and saves it as a .pkl file.

    Returns
    -------
    float
        Test-set RMSE of the final ensemble.
    """

    # ── Load processed data ──────────────────────────────────────────────────
    df = pd.read_csv(input_path)

    if "Date" in df.columns:
        df = df.drop(columns=["Date"])
    df = df.select_dtypes(include=[np.number])

    # ── Build target: next day's (scaled) Daily_Return ───────────────────────
    df["Target"] = df["Daily_Return"].shift(-1)
    df.dropna(inplace=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["Target"].values

    # ── 80 / 20 chronological split ──────────────────────────────────────────
    split   = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"🚀  Training ensemble on {len(X_train):,} samples "
          f"({len(available)} features)  ·  Test: {len(X_test):,} samples")

    # ─────────────────────────────────────────────────────────────────────────
    # Level-0 base learners
    # ─────────────────────────────────────────────────────────────────────────

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("   Training XGBoost …")
    xgb_model = xgb.XGBRegressor(
        n_estimators       = 2000,
        learning_rate      = 0.015,
        max_depth          = 6,
        subsample          = 0.80,
        colsample_bytree   = 0.80,
        colsample_bylevel  = 0.80,
        min_child_weight   = 5,
        gamma              = 0.05,
        reg_alpha          = 0.10,
        reg_lambda         = 1.50,
        random_state       = 42,
        n_jobs             = -1,
        tree_method        = "hist",
        early_stopping_rounds = 75,
        eval_metric        = "rmse",
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set           = [(X_test, y_test)],
        verbose            = False,
    )
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test))))
    print(f"      XGBoost test RMSE : {xgb_rmse:.6f}")

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgb_model = None
    if _HAS_LGB:
        print("   Training LightGBM …")
        lgb_model = lgb.LGBMRegressor(
            n_estimators     = 2000,
            learning_rate    = 0.015,
            max_depth        = 6,
            num_leaves       = 50,
            subsample        = 0.80,
            colsample_bytree = 0.80,
            min_child_samples= 20,
            reg_alpha        = 0.10,
            reg_lambda       = 1.50,
            random_state     = 42,
            n_jobs           = -1,
            verbose          = -1,
        )
        callbacks = [lgb.early_stopping(75, verbose=False), lgb.log_evaluation(-1)]
        lgb_model.fit(
            X_train, y_train,
            eval_set   = [(X_test, y_test)],
            callbacks  = callbacks,
        )
        lgb_rmse = float(np.sqrt(mean_squared_error(y_test, lgb_model.predict(X_test))))
        print(f"      LightGBM test RMSE: {lgb_rmse:.6f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("   Training RandomForest …")
    rf_model = RandomForestRegressor(
        n_estimators    = 600,
        max_depth       = 12,
        min_samples_leaf= 5,
        max_features    = "sqrt",
        random_state    = 42,
        n_jobs          = -1,
    )
    rf_model.fit(X_train, y_train)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_model.predict(X_test))))
    print(f"      RandomForest RMSE   : {rf_rmse:.6f}")

    # ── Extra Trees ───────────────────────────────────────────────────────────
    print("   Training ExtraTrees …")
    et_model = ExtraTreesRegressor(
        n_estimators    = 600,
        max_depth       = 12,
        min_samples_leaf= 5,
        max_features    = "sqrt",
        random_state    = 42,
        n_jobs          = -1,
    )
    et_model.fit(X_train, y_train)
    et_rmse = float(np.sqrt(mean_squared_error(y_test, et_model.predict(X_test))))
    print(f"      ExtraTrees RMSE     : {et_rmse:.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Level-1 stacking: Ridge meta-learner on OOF predictions
    # ─────────────────────────────────────────────────────────────────────────
    print("   Building stacking meta-learner …")

    base_models = [xgb_model, rf_model, et_model]
    base_names  = ["xgb",     "rf",     "et"]
    if lgb_model is not None:
        base_models.append(lgb_model)
        base_names.append("lgb")

    # Out-of-fold predictions on TRAINING set
    oof_preds = np.column_stack([
        _oof_predictions(m, X_train, y_train) for m in base_models
    ])

    # Test predictions from fully-trained base models
    test_preds = np.column_stack([
        m.predict(X_test) for m in base_models
    ])

    # Ridge meta-learner (fits on OOF, predicts on test_preds)
    meta = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(oof_preds, y_train)
    meta_weights = meta.coef_ / meta.coef_.sum()   # normalised weights
    meta_weights = np.clip(meta_weights, 0, None)  # non-negative
    if meta_weights.sum() == 0:
        meta_weights = np.ones(len(base_models)) / len(base_models)
    else:
        meta_weights /= meta_weights.sum()

    print(f"   Meta-learner weights: "
          + ", ".join(f"{n}={w:.3f}" for n, w in zip(base_names, meta_weights)))

    # ── Final ensemble ────────────────────────────────────────────────────────
    ensemble = WeightedEnsemble(models=base_models, weights=meta_weights.tolist())

    # ── Evaluate ensemble ─────────────────────────────────────────────────────
    y_pred = meta.predict(test_preds)   # stacked prediction
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae    = float(mean_absolute_error(y_test, y_pred))
    r2     = float(r2_score(y_test, y_pred))

    # Attach the meta-learner to the ensemble for prediction
    # We wrap everything so that ensemble.predict(X) goes through the full stack
    class StackedEnsemble:
        """Full stacking pipeline: base models → meta Ridge → prediction."""
        def __init__(self, base_models, meta_model):
            self.base_models = base_models
            self.meta_model  = meta_model

        def predict(self, X):
            level1 = np.column_stack([m.predict(X) for m in self.base_models])
            return self.meta_model.predict(level1)

    stacked = StackedEnsemble(base_models=base_models, meta_model=meta)

    print(f"\n✅  Stacked Ensemble → RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")
    print(f"   (XGB: {xgb_rmse:.4f}, RF: {rf_rmse:.4f}, ET: {et_rmse:.4f}"
          + (f", LGB: {lgb_rmse:.4f}" if lgb_model else "") + ")")

    # ── Save model ───────────────────────────────────────────────────────────
    # Use the path as given; replace .pt with .pkl if needed
    save_path = model_output_path.replace(".pt", ".pkl")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    joblib.dump(stacked, save_path)
    print(f"✅  Model saved → {save_path}")

    # Also write to the exact path requested (if different)
    if save_path != model_output_path:
        joblib.dump(stacked, model_output_path)

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "rmse":          rmse,
        "mae":           mae,
        "r2":            r2,
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "features":      available,
        "n_features":    len(available),
        "base_models":   base_names,
        "meta_weights":  {n: float(w) for n, w in zip(base_names, meta_weights)},
    }

    if metrics_path:
        os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅  Metrics saved → {metrics_path}")

    return rmse


# ── Standalone execution ──────────────────────────────────────────────────────
if __name__ == "__main__":
    _root        = os.path.join(os.path.dirname(__file__), "..")
    INPUT_FILE   = os.path.join(_root, "data", "processed", "processed_data.csv")
    MODEL_OUTPUT = os.path.join(_root, "models", "random_forest_new.pkl")
    METRICS_PATH = os.path.join(_root, "models", "metrics_new.json")

    train_model(INPUT_FILE, MODEL_OUTPUT, METRICS_PATH)
