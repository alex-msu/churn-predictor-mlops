from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    return metrics, y_pred