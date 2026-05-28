import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


def diagnosticar_regresion_gbm(df, target_col, test_size=0.2, random_state=42):
    feat_cols = [col for col in df.columns if col != target_col]
    X_df = df[feat_cols].copy()
    y_arr = df[target_col].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y_arr, test_size=test_size, random_state=random_state
    )

    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_tr, y_tr)

    rmse_train = round(float(np.sqrt(np.mean((model.predict(X_tr) - y_tr) ** 2))), 4)
    rmse_test  = round(float(np.sqrt(np.mean((model.predict(X_te) - y_te) ** 2))), 4)
    diagnostico = "overfitting" if rmse_test > 1.5 * rmse_train else "buen_ajuste"

    return {
        "rmse_train":  rmse_train,
        "rmse_test":   rmse_test,
        "diagnostico": diagnostico,
    }