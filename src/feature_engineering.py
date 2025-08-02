import numpy as np

def calcular_target(df):
    tx_cols = [f'Txs_T{i:02}' for i in range(1, 13) if f'Txs_T{i:02}' in df.columns]
    df = df.copy()
    df['total_transacciones_anual'] = df[tx_cols].sum(axis=1)
    df['total_transacciones_anual_log'] = np.log1p(df['total_transacciones_anual'])

    # Calcula umbral por mediana:
    umbral = df["total_transacciones_anual_log"].median()
    # Target binario usando ese umbral:
    df["target_binario"] = (df["total_transacciones_anual_log"] > umbral).astype(int)
    return df