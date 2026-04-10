import os

import numpy
from dotenv import load_dotenv
from flask import Flask, render_template, request
from models.bayesian import predict_bayesian
from models.dt import predict as predict_dt
from models.knn import predict_knn
from models.lasso_regression import predict as predict_lasso
from models.linear_svr import predict as predict_linear_svr
from models.linear import predict_linear
from models.rf import predict as predict_rf
from models.ridge import predict_ridge

load_dotenv()

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_cols = ["age"]
    float_cols = [
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
    string_cols = [
        "gender",
        "family_income",
        "parent_education",
        "device_type",
        "extracurriculars",
        "internet_access",
        "school_type"
    ]
    data = {}
    for col in int_cols:
        data[col] = int(request.form.get(col))
    for col in float_cols:
        data[col] = float(request.form.get(col))
    for col in string_cols:
        data[col] = request.form.get(col)
    result = {
        "classification": {
            "details": {
                "Naive Bayesian": predict_bayesian(data),
                "KNN": predict_knn(data),
                "Decision Tree": predict_dt(data),
                "Random Forest": predict_rf(data)
            }
        },
        "regression": {
            "details": {
                "Linear Regression": round(float(predict_linear(data)), 2),
                "Ridge Regression": round(float(predict_ridge(data)), 2),
                "Lasso Regression": round(float(predict_lasso(data)), 2),
                "Linear SVR": round(float(predict_linear_svr(data)), 2)
            }
        }
    }
    pass_count = 0
    fail_count = 0
    for classification_result in result["classification"]["details"].keys():
        if result["classification"]["details"][classification_result] == "Pass":
            pass_count += 1
        elif result["classification"]["details"][classification_result] == "Fail":
            fail_count += 1
    result["classification"]["general"] = "Fail" if fail_count > pass_count else "Pass"
    regression_count = 0
    predicted_value = 0
    for regression_result in result["regression"]["details"].keys():
        regression_count += 1
        predicted_value +=  round(float(result["regression"]["details"][regression_result]), 2)
    result["regression"]["general"] = round(predicted_value / regression_count, 2)
    return result


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
