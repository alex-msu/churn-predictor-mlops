import pandas as pd
import re
from pathlib import Path

def clasificar_columnas_mixtas(df, threshold=0.7):
    columnas_numericas = []
    columnas_categoricas = []

    for col in df.columns:
        valores = df[col].dropna().astype(str).str.strip()
        if len(valores) == 0:
            continue

        muestra = valores.sample(min(len(valores), 100), random_state=1)
        total = len(muestra)
        num_ok = 0
        cat_ok = 0

        for v in muestra:
            v_clean = v.replace(" ", "").replace(".", "").replace(",", ".")
            try:
                float(v_clean)
                num_ok += 1
            except ValueError:
                if len(v) <= 3 and v.upper() in {"R", "T", "P", "M", "F", "NULL", "NA", "NAN"}:
                    cat_ok += 1

    return columnas_numericas, columnas_categoricas