import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def medir_rendimiento_incremental(df, target_col):
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    selector = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
    selector.fit(X, y)

    ranking = selector.ranking_
    orden_idx = np.argsort(ranking)
    mejores_cols = [feature_cols[i] for i in orden_idx]

    r2_incremental = []
    for n in range(1, len(mejores_cols) + 1):
        cols_n = mejores_cols[:n]
        modelo = LinearRegression()
        modelo.fit(X[cols_n], y)
        pred = modelo.predict(X[cols_n])
        r2_incremental.append(r2_score(y, pred))

    return np.array(r2_incremental)