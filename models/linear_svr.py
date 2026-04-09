from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR


def train() -> Pipeline:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "student_performance_prediction_dataset-2.csv"
    model_dir = project_root / "pretrained"
    model_path = model_dir / "linearsvr_model.pkl"

    df = pd.read_csv(data_path)
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
                    dual=False,
                    random_state=42,
                    loss="squared_epsilon_insensitive",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r2:.4f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")

    return pipeline


if __name__ == "__main__":
    train()
