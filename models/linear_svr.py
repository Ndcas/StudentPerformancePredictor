import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

load_dotenv()

K_FOLDS = int(os.getenv("K_FOLDS", 5))
CPU_THREADS = int(os.getenv("CPU_THREADS", -1))


def train() -> Pipeline:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "student_performance_prediction_dataset-2.csv"
    model_dir = project_root / "pretrained"
    model_path = model_dir / "linearsvr_model.pkl"

    df = pd.read_csv(data_path, keep_default_na=False)
    df = df.drop(columns=["student_id", "grade_category", "pass_fail"])

    y = df["final_grade"]
    X = df.drop(columns=["final_grade"])

    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LinearSVR(
                    C=1.0,
                    epsilon=0.0,
                    tol=1e-4,
                    max_iter=10000,
                    fit_intercept=True,
                    loss="squared_epsilon_insensitive",
                    dual=False,
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "model__C": [0.1, 1.0, 10.0],
        "model__epsilon": [0.0, 0.5, 1.0],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        refit="mse",
        cv=K_FOLDS,
        n_jobs=CPU_THREADS,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    cv_results = pd.DataFrame(grid.cv_results_)
    comparison_cols = [
        "param_model__C",
        "param_model__epsilon",
        "mean_test_mae",
        "mean_test_mse",
        "mean_test_r2",
        "rank_test_mse",
    ]
    comparison_table = cv_results[comparison_cols].copy()
    comparison_table["mean_test_mae"] = -comparison_table["mean_test_mae"]
    comparison_table["mean_test_mse"] = -comparison_table["mean_test_mse"]
    comparison_table = comparison_table.sort_values("rank_test_mse")

    print("LinearSVR parameter comparison (CV):")
    print(comparison_table.to_string(index=False))
    print(f"Best params: {grid.best_params_}")

    y_pred = best_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    print(f"Saved model to: {model_path}")

    return best_pipeline


def predict(data: Any):
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "pretrained" / "linearsvr_model.pkl"
    pipeline = joblib.load(model_path)

    if isinstance(data, pd.DataFrame):
        input_df = data
    elif isinstance(data, list):
        input_df = pd.DataFrame(data)
    else:
        input_df = pd.DataFrame([data])

    return pipeline.predict(input_df)[0]


if __name__ == "__main__":
    train()
