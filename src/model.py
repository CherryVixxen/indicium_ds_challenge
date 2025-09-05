from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer

from .features import feature_groups, intersect_existing

def build_pipeline(df):
    num, cat, text, target = feature_groups()
    num = intersect_existing(df, num)
    cat = intersect_existing(df, cat)

    transformers = []
    if num:
        transformers.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]),
            num,
        ))
    if cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat,
        ))
    if text and text in df.columns:
        transformers.append(("txt", TfidfVectorizer(min_df=3, ngram_range=(1, 2)), text))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

    model = SGDRegressor(random_state=42, max_inter=1000, tol=1e-3, penalty="elasticnet", alpha=1e-4, l1_ratio=0.15)
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe, target

def train_validate(df, test_size: float = 0.2, seed: int = 42):
    pipe, target = build_pipeline(df)
    X = df.drop(columns=[target])
    y = df[target].astype(float)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=seed)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_va)

    rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
    mae = float(mean_absolute_error(y_va, preds))
    return rmse, mae, pipe