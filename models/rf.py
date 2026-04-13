import datetime
import os
from pathlib import Path

import joblib
import pandas
from pandas import DataFrame
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT.joinpath(os.getenv("DATA_PATH"))
SAVE_PATH = ROOT.joinpath(os.getenv("SAVE_PATH"))

MODEL_FILE_NAME = "rf.pkl"
ENCODER_FILE_NAME = "rf_encoder.pkl"


def train():
    start = datetime.datetime.now()
    print(f"Bắt đầu huấn luyện Random Forest lúc {start}")

    data = pandas.read_csv(DATA_PATH, keep_default_na=False)

    data = data.drop(["student_id", "final_grade", "grade_category"], axis=1)

    x = data.drop("pass_fail", axis=1)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["pass_fail"])

    categorical_cols = [
        "gender", "family_income", "parent_education",
        "device_type", "extracurriculars"
    ]

    binary_cols = {
        "internet_access": ["Yes", "No"],
        "school_type": ["Public", "Private"]
    }

    transformers = []

    for col in categorical_cols:
        transformers.append((col, OneHotEncoder(handle_unknown="ignore"), [col]))

    for col in binary_cols:
        transformers.append((col, OrdinalEncoder(categories=[binary_cols[col]]), [col]))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=1, n_jobs=-1))
    ])

    param_dist = {
        "classifier__n_estimators": [30, 50, 80],
        "classifier__max_depth": [5, 10, 15],
        "classifier__min_samples_split": [5, 10],
        "classifier__min_samples_leaf": [3, 5, 10],
        "classifier__max_features": ["sqrt"]
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        verbose=2,
        random_state=1
    )

    print("Đang huấn luyện Random Forest...")
    search.fit(x_train, y_train)

    print("\n===== Kết quả từng iteration =====")

    results = pandas.DataFrame(search.cv_results_)

    for i in range(len(results)):
        params = results.loc[i, "params"]
        f1 = results.loc[i, "mean_test_score"]

        # Train lại model với param này để tính accuracy trên test set
        model = pipeline.set_params(**params)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nIteration {i + 1}:")
        print(f"Params: {params}")
        print(f"F1 (CV): {f1:.4f}")
        print(f"Accuracy (test): {acc:.4f}")

    print("\nBest params:")
    for param in search.best_params_:
        print(f"{param}: {search.best_params_[param]}")

    print(f"F1: {search.best_score_:.4f}")

    y_pred = search.best_estimator_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")

    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall:.4f}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    joblib.dump(search.best_estimator_, SAVE_PATH / MODEL_FILE_NAME)
    joblib.dump(label_encoder, SAVE_PATH / ENCODER_FILE_NAME)

    joblib.dump(label_encoder, SAVE_PATH / ENCODER_FILE_NAME)

    print(f"Hoàn tất trong {datetime.datetime.now() - start}")


def predict(data):
    model = joblib.load(SAVE_PATH / MODEL_FILE_NAME)
    encoder = joblib.load(SAVE_PATH / ENCODER_FILE_NAME)

    input_df = DataFrame([data])

    y_pred = model.predict(input_df)
    return encoder.inverse_transform(y_pred)[0]


if __name__ == "__main__":
    train()