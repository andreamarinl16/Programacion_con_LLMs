import pandas as pd
import numpy as np


def deduplicar_pacientes(df, col_id, col_fecha, cols_clinicas):
    df = df.copy()

    df['score_completitud'] = df[cols_clinicas].notna().sum(axis=1)

    df[col_fecha] = pd.to_datetime(df[col_fecha])

    df_sorted = df.sort_values(
        by=[col_id, col_fecha, 'score_completitud'],
        ascending=[True, False, False]
    )

    df_dedup = df_sorted.groupby(col_id, sort=False).first().reset_index()

    return df_dedup