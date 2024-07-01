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

@app.route('/predict', methods = ["POST"])
def predict():
    
    data = request.json['data']
    print("////////////////////////////////////////")
    specs = []
    for key in data:
        specs.append(data[key])
    
    specs = np.array(specs)
    
    return jsonify(np.exp(model.predict(specs.reshape(1,12))[0]))


if __name__ == "__main__":
    app.run(debug=True)

