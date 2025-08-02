from src.load_data import descargar_dataset, cargar_datos
from src.preprocessing import limpiar_columnas, eliminar_outliers, escalar_variables
from src.feature_engineering import calcular_target
from src.model_training import dividir_datos, entrenar_modelo, guardar_modelo
from src.evaluation import evaluar_modelo
from datetime import datetime

import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
import argparse

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve,
    classification_report
)

mlflow.end_run()  # Cierra cualquier run residual
mlflow.set_tracking_uri('http://localhost:5000') # Asegurarse de ejecutar el servidor MLflow
mlflow.set_experiment("Churn-Predictor-MLops")

parser = argparse.ArgumentParser(description="Entrenamiento modelo churn con MLflow")
parser.add_argument("--backup", action="store_true", help="Guardar copia local del modelo con joblib")
parser.add_argument("--verbose", type=int, default=0, help="Nivel de verbosidad del entrenamiento")
parser.add_argument("--importancia", action="store_true", help="Mostrar importancia de características")
parser.add_argument("--n-estimators", type=int, default=100, help="Cantidad de árboles en el modelo")
parser.add_argument("--max-depth", type=int, default=None, help="Máxima profundidad de los árboles")
args = parser.parse_args()

def guardar_metricas_visuales(modelo, X_test, y_test, y_pred, carpeta='artifacts/metrics'):
    os.makedirs(carpeta, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    ruta_cm = os.path.join(carpeta, f"conf_matrix_{timestamp}.png")
    plt.savefig(ruta_cm)
    plt.close()
    mlflow.log_artifact(ruta_cm)

    # 2. Curvas ROC y Precision-Recall (si el modelo lo permite)
    if hasattr(modelo, "predict_proba"):
        y_proba = modelo.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title('ROC Curve')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        ruta_roc = os.path.join(carpeta, f"roc_curve_{timestamp}.png")
        plt.savefig(ruta_roc)
        plt.close()
        mlflow.log_artifact(ruta_roc)

        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        ruta_pr = os.path.join(carpeta, f"pr_curve_{timestamp}.png")
        plt.savefig(ruta_pr)
        plt.close()
        mlflow.log_artifact(ruta_pr)
    else:
        print("[WARN] El modelo no soporta predict_proba, no se generan curvas ROC / PR.")

    # 3. Importancia de características
    try:
        importances = modelo.named_steps["classifier"].feature_importances_
        feature_names = modelo.named_steps["preprocessor"].get_feature_names_out()
        importancia_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Gráfico
        top_n = min(20, len(importancia_df))
        plt.figure(figsize=(10, 6))
        plt.barh(importancia_df["feature"][:top_n][::-1], importancia_df["importance"][:top_n][::-1])
        plt.title("Importancia de características (top 20)")
        plt.tight_layout()
        ruta_feat = os.path.join(carpeta, f"feature_importance_{timestamp}.png")
        plt.savefig(ruta_feat)
        plt.close()
        mlflow.log_artifact(ruta_feat)

        # CSV
        ruta_csv = os.path.join(carpeta, f"feature_importance_{timestamp}.csv")
        importancia_df.to_csv(ruta_csv, index=False)
        mlflow.log_artifact(ruta_csv)

    except Exception as e:
        print(f"[WARN] No se pudo exportar importancia de características: {e}")

    # 4. Distribución del target
    counts = pd.Series(y_test).value_counts().sort_index()
    plt.figure(figsize=(4, 4))
    counts.plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Distribución del target (y_test)")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=0)
    plt.tight_layout()
    ruta_dist = os.path.join(carpeta, f"target_distribution_{timestamp}.png")
    plt.savefig(ruta_dist)
    plt.close()
    mlflow.log_artifact(ruta_dist)
    print(f"[INFO] Visualizaciones guardadas en: {carpeta}")

def main():
    print("[INFO] Iniciando pipeline...")

    # A. Cargar los datos
    print("[INFO] Descargando y cargando datos...")
    path = descargar_dataset()
    df = cargar_datos(path)

    # B. Limpieza y transformación inicial
    print("[INFO] Limpiando columnas y corrigiendo tipos...")
    df = limpiar_columnas(df)

    print("[INFO] Calculando variable target...")
    df = calcular_target(df)

    print("/!\ Eliminando outliers...")
    outlier_cols = ['Renta', 'CUPO_L1', 'CUPO_L2', 'CUPO_MX']
    df = eliminar_outliers(df, outlier_cols)

    print("[INFO] Estandarizando y normalizando variables...")
    tx_cols = [f'Txs_T{i:02}' for i in range(1, 13) if f'Txs_T{i:02}' in df.columns]
    std_cols = ['Renta', 'CUPO_L1', 'CUPO_L2', 'CUPO_MX']
    df = escalar_variables(df, std_cols, tx_cols)

    # C. División de datos y entrenamiento
    print("[INFO] Dividiendo datos y entrenando modelo...")
    X_train, X_test, y_train, y_test = dividir_datos(df)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['total_transacciones_anual', 'total_transacciones_anual_log']).tolist()

    # Elimina target de los conjuntos de columnas
    if "target_binario" in num_cols:
        num_cols.remove("target_binario")
    if "target_binario" in cat_cols:
        cat_cols.remove("target_binario")

    # Eliminar columnas categóricas con más de 100 niveles
    cat_cols = [col for col in cat_cols if df[col].nunique() <= 100]

    # Verificar si ya existe modelo entrenado
    modelo_path = "models/modelo_rf.pkl"
    if os.path.exists(modelo_path):
        decision = input(f"/!\ Ya existe un modelo entrenado en '{modelo_path}'. ¿Deseas cargarlo? (s/n): ").lower().strip()
        if decision == "s":
            print("[INFO] Cargando modelo previamente entrenado...")
            cargado = joblib.load(modelo_path)
            modelo = cargado["modelo"]
            input_example = cargado["input_example"]
            signature = cargado["signature"]
            entrenar = False
        else:
            print("[INFO] Entrenando modelo desde cero...")
            entrenar = True
    else:
        entrenar = True

    if entrenar:
        modelo, input_example, signature = entrenar_modelo(
            X_train, y_train,
            cat_cols, num_cols,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            verbose=args.verbose
        )

    if args.importancia:
        try:
            import pandas as pd
            importances = modelo.named_steps["classifier"].feature_importances_
            feature_names = modelo.named_steps["preprocessor"].get_feature_names_out()
            importancia_df = pd.DataFrame({
                "feature": feature_names,
                "importancia": importances
            }).sort_values(by="importancia", ascending=False)
            print("\n[INFO] Importancia de características:")
            print(importancia_df.head(20).to_string(index=False))
        except Exception as e:
            print(f"/!\ No se pudo calcular la importancia de características: {e}")

    # D. Evaluación con MLflow
    with mlflow.start_run(run_name="RandomForest_Churn_Clasificacion"):
        mlflow.set_tag("author", "alex-msu")

        # Registrar parámetros
        mlflow.log_param("modelo", "RandomForestClassifier")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # Loguear modelo con firma
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo_rf",
            input_example=input_example,
            signature=signature
        )

        print("[INFO] Evaluando modelo...")
        resultados, y_pred = evaluar_modelo(modelo, X_test, y_test)

        print("[INFO] Métricas de evaluación:")
        for k, v in resultados.items():
            mlflow.log_metric(k, v)

        guardar_metricas_visuales(modelo, X_test, y_test, y_pred)

        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # E. Guardar modelo
        if entrenar and args.backup:
            print("[SAVE] Guardando modelo entrenado...")
            os.makedirs("models", exist_ok=True)
            guardar_modelo(modelo, input_example, signature)

    print("[SUCCESS] Pipeline completo.")

if __name__ == "__main__":
    main()
