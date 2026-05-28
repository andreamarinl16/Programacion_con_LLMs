import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocesar_datos_clinicos(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    X_num = X.select_dtypes(include=[np.number])
    X_cat = X.select_dtypes(exclude=[np.number])

    imputer_num = SimpleImputer(strategy="mean")
    X_num_imputada = imputer_num.fit_transform(X_num)

    scaler = StandardScaler()
    X_num_escalada = scaler.fit_transform(X_num_imputada)

    imputer_cat = SimpleImputer(strategy="most_frequent")
    X_cat_imputada = imputer_cat.fit_transform(X_cat)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_codificada = encoder.fit_transform(X_cat_imputada)

    X_final = np.hstack([X_num_escalada, X_cat_codificada])

    return X_final, y