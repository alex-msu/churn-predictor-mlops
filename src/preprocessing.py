import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def limpiar_columnas(df):

    for col in df.columns:
        try:
            # Intentamos convertir a numérico de forma segura
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # Conversión de columnas numéricas mal tipadas
    columnas_convertir = ['Renta', 'CUPO_L1', 'CUPO_L2', 'CUPO_MX']
    for col in columnas_convertir:
        if col in df.columns:
            # Limpieza personalizada si hay separadores de miles, comas, símbolos
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Imputaciones básicas
    if 'Sexo' in df.columns:
        df['Sexo'] = df['Sexo'].fillna(df['Sexo'].mode()[0])
    if 'Region' in df.columns:
        df['Region'] = df['Region'].fillna(df['Region'].mode()[0])
        df['Region'] = df['Region'].astype('category')

    # Imputación de renta por subsegmento
    if 'Renta' in df.columns and 'Subsegmento' in df.columns:
        df['Renta'] = df.groupby('Subsegmento')['Renta'].transform(lambda x: x.fillna(x.median()))

    # Imputación de CambioPin (si existe)
    if 'CambioPin' in df.columns:
        df['CambioPin'] = df['CambioPin'].fillna(df['CambioPin'].median())

    # Forzar consistencia de tipos en columnas categóricas
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].astype(str)

    # 🔧 Imputar cualquier valor numérico restante con la mediana general (último filtro)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # 🔧 (Opcional) Imputar cualquier categoría con 'desconocido' si llegara a colarse algo
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna("desconocido")

    return df


def eliminar_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

def escalar_variables(df, cols_std, cols_norm):
    df[cols_std] = StandardScaler().fit_transform(df[cols_std])
    df[cols_norm] = MinMaxScaler().fit_transform(df[cols_norm])
    return df
