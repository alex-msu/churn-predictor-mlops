from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import mlflow.models

def dividir_datos(df, target_col='total_transacciones_anual_log'):
    X = df.drop(columns=['total_transacciones_anual', 'total_transacciones_anual_log', 'target_binario'])
    y = df["target_binario"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convertir enteros con NaNs a float64
    for col in X_train.select_dtypes(include='integer').columns:
        if X_train[col].isna().any():
            X_train[col] = X_train[col].astype('float64')
            X_test[col] = X_test[col].astype('float64')

    return X_train, X_test, y_train, y_test


def entrenar_modelo(X_train, y_train, cat_cols, num_cols, n_estimators=100, max_depth=None, verbose=0):
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=verbose
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Genera input_example y firma del modelo
    input_example = X_train.head(1).copy()

    # Convertimos TODOS los int a float64 por seguridad ante posibles NaNs futuros
    int_cols = input_example.select_dtypes(include="integer").columns
    input_example[int_cols] = input_example[int_cols].astype("float64")

    signature = mlflow.models.infer_signature(input_example, pipeline.predict(input_example))

    return pipeline, input_example, signature

def guardar_modelo(modelo, input_example, signature, ruta='models/modelo_rf.pkl'):
    joblib.dump({
        "modelo": modelo,
        "input_example": input_example,
        "signature": signature
    }, ruta)