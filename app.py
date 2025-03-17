from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained models
model_slr = pickle.load(open("model_slr.pkl", "rb"))
model_mlr = pickle.load(open("model_mlr.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/slr")
def slr_page():
    return render_template("slr_prediction.html", prediction=None)

@app.route("/mlr")
def mlr_page():
    return render_template("mlr_prediction.html", prediction=None)

@app.route("/predict_slr", methods=["POST"])
def predict_slr():
    try:
        student_mark = float(request.form["student_mark"])
        prediction = model_slr.predict(np.array([[student_mark]]))[0]
        return render_template("slr_prediction.html", prediction=round(prediction, 2), entered_value=student_mark)
    except ValueError:
        return render_template("slr_prediction.html", error="Invalid input. Please enter a valid number.")

@app.route("/predict_mlr", methods=["POST"])
def predict_mlr():
    try:
        student_mark = float(request.form["student_mark"])
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])

        # Predict using MLR model
        input_features = np.array([[student_mark, study_hours, attendance]])
        prediction = model_mlr.predict(input_features)[0]

        return render_template("mlr_prediction.html", prediction=round(prediction, 2),
                               entered_mark=student_mark, entered_hours=study_hours, entered_attendance=attendance)

    except ValueError:
        return render_template("mlr_prediction.html", error="Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    app.run(debug=True)
