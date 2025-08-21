from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

# List of expected feature names in the correct order
num_attribs = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income'
]
cat_attribs = ['ocean_proximity']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Collect input values from the form
        values = []
        for col in num_attribs:
            values.append(float(request.form[col]))
        # Categorical value
        cat_value = request.form['ocean_proximity']
        df = pd.DataFrame([values + [cat_value]], columns=num_attribs + cat_attribs)

        # Transform and predict
        X_prepared = pipeline.transform(df)
        prediction = model.predict(X_prepared)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
