import datetime
import os
from pathlib import Path

import joblib
import pandas
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT.joinpath(os.getenv("DATA_PATH"))
SAVE_PATH = ROOT.joinpath(os.getenv("SAVE_PATH"))
K_FOLDS = int(os.getenv("K_FOLDS"))
CPU_THREADS = int(os.getenv("CPU_THREADS"))

MODEL_FILE_NAME = "dt.pkl"
ENCODER_FILE_NAME = "dt_encoder.pkl"


def train():
    start = datetime.datetime.now()
    print(f"Bắt đầu huấn luyện Decision Tree lúc {start}")

    data = pandas.read_csv(DATA_PATH)
    data = data.drop(["student_id", "final_grade", "grade_category"], axis=1)

    x = data.drop("pass_fail", axis=1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["pass_fail"])

    numeric_cols = [
        "age", "study_hours", "attendance", "sleep_hours",
        "previous_grade", "assignments_completed", "practice_tests_taken",
        "group_study_hours", "notes_quality_score", "time_management_score",
        "motivation_level", "mental_health_score", "screen_time", "social_media_hours"
    ]

    categorical_cols = [
        "gender", "family_income", "parent_education",
        "device_type", "extracurriculars"
    ]

    binary_cols = {
        "internet_access": ["Yes", "No"],
        "school_type": ["Public", "Private"]
    }

    transformers = []

    for col in numeric_cols:
        transformers.append((col, StandardScaler(), [col]))

    for col in categorical_cols:
        transformers.append((col, OneHotEncoder(handle_unknown="ignore"), [col]))

    for col in binary_cols:
        transformers.append((col, OrdinalEncoder(categories=[binary_cols[col]]), [col]))

    preprocessor = ColumnTransformer(transformers=transformers)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=1))
    ])

    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": [None, "sqrt", "log2"]
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=1, stratify=y
    )

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=K_FOLDS,
        scoring="f1",
        n_jobs=CPU_THREADS,
        verbose=4
    )

    print("Đang huấn luyện Decision Tree...")
    grid.fit(x_train, y_train)

    print("Best params:")
    for param in grid.best_params_:
        print(f"{param}: {grid.best_params_[param]}")

    print(f"F1: {grid.best_score_:.4f}")

    y_pred = grid.best_estimator_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    joblib.dump(grid.best_estimator_, SAVE_PATH / MODEL_FILE_NAME)
    joblib.dump(label_encoder, SAVE_PATH / ENCODER_FILE_NAME)

    print(f"Hoàn tất trong {datetime.datetime.now() - start}")


def predict(data):
    model = joblib.load(SAVE_PATH / MODEL_FILE_NAME)
    encoder = joblib.load(SAVE_PATH / ENCODER_FILE_NAME)

    y_pred = model.predict(data)
    return encoder.inverse_transform(y_pred)


if __name__ == "__main__":
    train()