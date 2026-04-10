from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train() -> Pipeline:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "student_performance_prediction_dataset-2.csv"
    model_dir = project_root / "pretrained"
    model_path = model_dir / "lasso_model.pkl"

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
                Lasso(
                    alpha=1.0,
                    fit_intercept=True,
                    max_iter=10000,
                    tol=1e-4,
                    selection="cyclic",
                    positive=False,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_
    zero_feature_names = [
        feature_names[idx]
        for idx, coef in enumerate(coefficients)
        if np.isclose(coef, 0.0)
    ]

    print("Zero-coefficient features:")
    if zero_feature_names:
        for feature_name in zero_feature_names:
            print(f"- {feature_name}")
    else:
        print("- None")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")

    return pipeline


def predict(data: Any):
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "pretrained" / "lasso_model.pkl"
    pipeline = joblib.load(model_path)

    if isinstance(data, pd.DataFrame):
        input_df = data
    elif isinstance(data, list):
        input_df = pd.DataFrame(data)
    else:
        input_df = pd.DataFrame([data])

    return pipeline.predict(input_df)


if __name__ == "__main__":
    train()
