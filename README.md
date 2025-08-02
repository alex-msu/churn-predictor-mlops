# 🧠 Predicción de Churn Bancario con MLflow y MLOps

Este proyecto implementa un flujo completo de machine learning aplicado a clientes bancarios, utilizando prácticas de MLOps. Se automatiza desde la carga y limpieza de datos hasta la ingeniería de características, entrenamiento del modelo, evaluación y trazabilidad con MLflow.

> 🔍 **Objetivo:** Predecir la retención o abandono de clientes bancarios a partir de comportamiento transaccional histórico y características demográficas.
---

## 🗂️ Estructura del Proyecto

```bash
churn-predictor-dormammu/
│
├── notebooks/                
│   └── demo.ipynb                  # Notebook de demostración (Jupyter)
│
├── src/                            # Código fuente modular
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils/
│       └── column_analysis.py
│
├── models/                         # Modelos entrenados (si se activa --backup)
│
├── images/                         # Gráficos para README o notebook
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py                         # Punto de entrada de la pipeline
````

---
## 📊 Dataset Utilizado

* 📦 Fuente: [Kaggle - Base Clientes Monopoly](https://www.kaggle.com/datasets/nadiaarellanog/base-clientes-monopoly)
* 👥 450.000+ clientes bancarios
* 📅 Variables temporales mensuales (últimos 12 meses)
* 🔢 Variables de uso, facturación, pagos, cobranza, etc.
* 🎯 Target generado: `target_binario` (0 = Retención, 1 = Abandono)

---
## ⚙️ Tecnologías utilizadas

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=for-the-badge&&logo=python&logoColor=white) ![Joblib](https://img.shields.io/badge/-Joblib-FFFFFF?style=for-the-badge&logo=Bastyon&logoColor=black)


[//]: # "- Python 3.10+"
[//]: # "- Jupyter"
[//]: # "- pandas, numpy"
[//]: # "- scikit-learn"
[//]: # "- mlflow"
[//]: # "- joblib"
[//]: # "- matplotlib, seaborn"

---

## 🚀 Ejecución del Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/alex-msu/churn-predictor-dormammu.git
cd churn-predictor-dormammu
```

### 2. Crear entorno virtual (opcional pero recomendado)

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Iniciar el servidor local de MLflow (seguimiento de experimentos)

```bash
mlflow server ^
  --backend-store-uri sqlite:///mlflow.db ^
  --default-artifact-root file:///C:/Users/TU_USUARIO/churn-predictor-dormammu/mlruns ^
  --host 127.0.0.1 ^
  --port 5000
```

Esto habilita el panel en: [http://localhost:5000](http://localhost:5000)

### 5. Ejecutar la pipeline

```bash
python main.py
```

---

## 🧾 Explicación de argumentos

| Argumento        | Descripción                                                       |
| ---------------- | ----------------------------------------------------------------- |
| `--backup`       | Guarda una copia del modelo entrenado en `models/` (formato .pkl) |
| `--importancia`  | Muestra la importancia de características tras el entrenamiento   |
| `--verbose`      | Nivel de detalle del entrenamiento (0 = silencioso)               |
| `--n-estimators` | Cantidad de árboles para el Random Forest                         |
| `--max-depth`    | Profundidad máxima de cada árbol                                  |

Ejemplo:

```bash
python main.py --backup --importancia --verbose 1 --n-estimators 200 --max-depth 10
```
Ejecutaría el pipeline guardando una copia del modelo, mostrando la importancia de características, con un nivel de detalle de entrenamiento 1, con 200 árboles para el Random Forest y una profundad máxima para cada árbol de 10.

---

---
## 📓 Notebook Demo (opcional)

```bash
cd notebooks/
jupyter notebook demo.ipynb
```

Incluye visualizaciones y ejecución por etapas para demostraciones o validación rápida.

---

## 🖼️ Resultados y Visualizaciones

### 🎯 Matriz de Confusión
![Matriz de confusión](images/conf_matrix_example.png)

### 📊 Importancia de Variables
![Importancia](images/feature_importance_example.png)

### 📈 Interfaz de MLflow (tracking server)
![MLflow UI](images/mlflow_ui.png)

### ⚖️ Distribución del Target
![Distribución del target](images/target_distribution_example.png)

*(Estas imágenes se generan automáticamente al ejecutar `main.py` y se guardan en `artifacts/metrics/`)*

---

## ✅ Resultados

* 🔍 **Evaluación con clasificación binaria:**

  * Precision, Recall, F1-score (registradas en MLflow) = ~0.94
* 🧠 **Modelo base:** RandomForestClassifier
* 🧪 **Registro automático en MLflow:** Modelo, métricas, artefactos y firma


---

## 🧠 MLflow en acción

* 🔎 Registro automático de métricas, parámetros y modelo
* 🧾 Firma automática con `infer_signature`
* 📁 Seguimiento persistente en `mlruns/`

---


## 🧼 Prácticas MLOps aplicadas

* Modularización en archivos fuente (`src/`)
* Logging de errores y trazabilidad (`debug_log.txt`)
* Preprocesamiento robusto de columnas mixtas
* Validación de tipos y estructura del dataset
* Registro explícito del modelo (`joblib`) y trazabilidad (`MLflow`)


---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

---

## 👥 Autor

Desarrollado por **Alexis Martínez**

> Proyecto académico con fines de aprendizaje en MLOps y clasificación supervisada en contextos financieros (entornos productivos).
