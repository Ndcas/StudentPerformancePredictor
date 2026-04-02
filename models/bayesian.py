import datetime
import math
import os
from pathlib import Path

import joblib
import pandas
from dotenv import load_dotenv
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT.joinpath(os.getenv("DATA_PATH"))
SAVE_PATH = ROOT.joinpath(os.getenv("SAVE_PATH"))
K_FOLDS = int(os.getenv("K_FOLDS"))
CPU_THREADS = int(os.getenv("CPU_THREADS"))
MODEL_FILE_NAME = "bayesian.pkl"
ENCODER_FILE_NAME = "bayesian_encoder.pkl"


def train():
    start = datetime.datetime.now()
    print(f"Bắt đầu quá trình huấn luyện mô hình Naive Bayesian lúc {start}")
    print("Đang đọc dữ liệu...")
    data = pandas.read_csv(DATA_PATH)
    data = data.drop(["student_id", "final_grade", "grade_category"], axis=1)
    x = data.drop("pass_fail", axis=1)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["pass_fail"])
    numeric_cols = [
        "age",
        "study_hours",
        "attendance",
        "sleep_hours",
        "previous_grade",
        "assignments_completed",
        "practice_tests_taken",
        "group_study_hours",
        "notes_quality_score",
        "time_management_score",
        "motivation_level",
        "mental_health_score",
        "screen_time",
        "social_media_hours"
    ]
    categorical_cols = [
        "gender",
        "family_income",
        "parent_education",
        "device_type",
        "extracurriculars"
    ]
    binary_cols = {
        "internet_access": ["Yes", "No"],
        "school_type": ["Public", "Private"]
    }
    param_grid = {}
    transformers = []
    for col in numeric_cols:
        if col == "age":
            transformers.append((col, "drop", [col]))
            continue
        transformers.append((col, StandardScaler(), [col]))
    for col in categorical_cols:
        if col == "parent_education":
            transformers.append((col, "drop", [col]))
            continue
        transformers.append((col, OneHotEncoder(), [col]))
    for col in binary_cols:
        transformers.append((col, OrdinalEncoder(categories=[binary_cols[col]]), [col]))
    preprocessor = ColumnTransformer(transformers=transformers)
    param_grid["classifier__var_smoothing"] = [10 ** i for i in range(-9, 1)]
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=K_FOLDS,
        scoring="f1",
        n_jobs=CPU_THREADS,
        verbose=4
    )
    print("Đang dò tham số và huấn luyện mô hình Naive Bayesian...")
    grid.fit(x_train, y_train)
    for param in grid.best_params_:
        print(f"{param}: {grid.best_params_[param]}")
    print(f"F1: {grid.best_score_:4f}")
    accuracy = accuracy_score(y_test, grid.best_estimator_.predict(x_test))
    print(f"Accuracy: {accuracy:4f}")
    print(f"Đang lưu mô hình")
    os.makedirs(SAVE_PATH, exist_ok=True)
    joblib.dump(grid.best_estimator_, SAVE_PATH.joinpath(MODEL_FILE_NAME))
    joblib.dump(label_encoder, SAVE_PATH.joinpath(ENCODER_FILE_NAME))
    print(f"Hoàn tất huấn luyện mô hình Naive Bayesian trong {datetime.datetime.now() - start}")


def predict_bayesian(data):
    pipeline = joblib.load(SAVE_PATH.joinpath(MODEL_FILE_NAME))
    encoder = joblib.load(SAVE_PATH.joinpath(ENCODER_FILE_NAME))
    input_df = DataFrame([data])
    result = encoder.inverse_transform(pipeline.predict(input_df))[0]
    return result


if __name__ == "__main__":
    train()
