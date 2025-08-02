import pandas as pd
import kagglehub
from src.schema_dtypes import forced_dtypes
from src.utils.column_analysis import clasificar_columnas_mixtas

from pathlib import Path

def descargar_dataset():
    path = kagglehub.dataset_download("nadiaarellanog/base-clientes-monopoly")
    return path

def cargar_datos(path):
    csv_path = f"{path}/Base_clientes_Monopoly.csv"

    try:
        df = pd.read_csv(
            csv_path,
            delimiter="\t",
            dtype=str,
            na_values=["?", "NA", "null", "", "None"],
            low_memory=False
        )
        print("[INFO] Datos cargados como texto para preprocesamiento.")

        numericas, categoricas = clasificar_columnas_mixtas(df)

        # 🔹 LIMPIEZA de categóricas ANTES que nada
        for col in categoricas:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                df[col] = df[col].replace({"NULL": pd.NA, "NAN": pd.NA})

        # 🔹 LIMPIEZA y conversión de columnas numéricas
        for col in numericas:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(".", "", regex=False)
                df[col] = df[col].str.replace(",", ".", regex=False)
                df[col] = pd.to_numeric(df[col], errors="coerce")

    except Exception as e:
        print(f"[ERROR] Falló la carga: {e}")
        raise e

    return df