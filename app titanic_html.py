# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, request, render_template
import joblib
import sys

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    n_features = []
    for i in request.form.values():
        n_features.append(i)
    cols = ['age', 'sex', 'embarked']
    print(n_features)
    features = pd.DataFrame([n_features],columns=cols)
    print(features)
    trans_data= pipeline.transform(features)
    print(trans_data)
    scaled_df = scalar.transform(trans_data)
    prediction = lr.predict(scaled_df)
    print(prediction)
    return render_template("result.html", prediction = prediction[0] )

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('C:\\Users\\GupAshis\Downloads\\Titanic_project\\Titanic_project\\my_model_titanic.pkl') # Load "model.pkl"
    print ('Model loaded')
    pipeline = joblib.load('C:\\Users\\GupAshis\Downloads\\Titanic_project\\Titanic_project\\my_pipe_titanic.pkl')
    print ('Pipeline loaded')
    scalar = joblib.load('C:\\Users\\GupAshis\Downloads\\Titanic_project\\Titanic_project\\my_scalar_titanic.pkl')
    print ('scalar loaded')
    
    
    app.run(port=port, debug=True)