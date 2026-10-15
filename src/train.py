"""Offline LightGBM training on chronological train rows."""
from __future__ import annotations
import numpy as np


def train_lightgbm(train_frame, valid_frame, columns, config=None, seed=42, num_threads=4):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is required for real-data training. Install requirements.txt in an offline-ready environment.") from exc
    cfg = {"objective":"regression", "n_estimators":500, "learning_rate":0.05, "num_leaves":63,
           "subsample":0.8, "colsample_bytree":0.8, "random_state":seed, "n_jobs":num_threads,
           "verbosity":-1}
    cfg.update(config or {})
    model = lgb.LGBMRegressor(**cfg)
    cat = [c for c in columns if str(train_frame[c].dtype) == "category"]
    model.fit(train_frame[columns], np.log1p(train_frame.sales), categorical_feature=cat,
              eval_set=[(valid_frame[columns], np.log1p(valid_frame.sales))], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model
