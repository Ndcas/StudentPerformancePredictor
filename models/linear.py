import datetime
import os
from pathlib import Path

import joblib
import pandas
from dotenv import load_dotenv
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT.joinpath(os.getenv("DATA_PATH", "student_performance_prediction_dataset-2.csv"))
SAVE_PATH = ROOT.joinpath(os.getenv("SAVE_PATH", "models"))
K_FOLDS = int(os.getenv("K_FOLDS", 5))
CPU_THREADS = int(os.getenv("CPU_THREADS", -1))
MODEL_FILE_NAME = "linear_regression.pkl"

def train():
    start = datetime.datetime.now()
    print(f"Bắt đầu quá trình huấn luyện mô hình Linear Regression lúc {start}")
    print("Đang đọc dữ liệu...")
    data = pandas.read_csv(DATA_PATH, keep_default_na=False)
    y = data["final_grade"]
    x = data.drop(["student_id", "pass_fail", "grade_category", "final_grade"], axis=1)
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
    
    param_grid["regressor__fit_intercept"] = [True, False]
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression(n_jobs=CPU_THREADS))
    ])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=K_FOLDS,
        scoring="r2", 
        n_jobs=CPU_THREADS,
        verbose=4
    )
    
    print("Đang dò tham số và huấn luyện mô hình Linear Regression...")
    grid.fit(x_train, y_train)
    
    for param in grid.best_params_:
        print(f"{param}: {grid.best_params_[param]}")
        
    print(f"R2 Score (Cross-Validation): {grid.best_score_:4f}")
    
    y_pred = grid.best_estimator_.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Test R2 Score: {r2:4f}")
    print(f"Test MSE: {mse:4f}")
    
    print(f"Đang lưu mô hình...")
    os.makedirs(SAVE_PATH, exist_ok=True)
    joblib.dump(grid.best_estimator_, SAVE_PATH.joinpath(MODEL_FILE_NAME))
    
    print(f"Hoàn tất huấn luyện mô hình Linear Regression trong {datetime.datetime.now() - start}")

def predict_linear(data):
    pipeline = joblib.load(SAVE_PATH.joinpath(MODEL_FILE_NAME))
    input_df = DataFrame([data])
    result = pipeline.predict(input_df)[0]
    return result

if __name__ == "__main__":
    train()