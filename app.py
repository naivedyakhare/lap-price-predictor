import pickle as pk
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# Create App
app = Flask(__name__)

# Load the model
model = pk.load(open("pipe.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    
    data = [request.form[key] for key in request.form]
    
    
    data[4] = "0" if data[4] == "No" else "1"
    data[5] = "0" if data[4] == "No" else "1"

    for i in range(len(data)):
        if data[i].isnumeric():
            data[i] = float(data[i])

    
    final_data = np.array(data).reshape(1,12)

    output = np.exp(model.predict(final_data))


    return render_template("index.html", prediction_text = f"The price of the laptop is {output[0]}")


if __name__ == "__main__":
    app.run(debug=True)

