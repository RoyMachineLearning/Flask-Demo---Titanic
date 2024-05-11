# -*- coding: utf-8 -*-

import pandas as pd
from flask import Flask, request, render_template
import joblib
import sys
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    n_features = []
    for i in request.form.values():
        n_features.append(i)
    cols = ['age', 'sex', 'pclass','embarked']
    print(n_features)
    features = pd.DataFrame([n_features],columns=cols)
    print(features)
    features['pclass'] = features['pclass'].astype(int)
    trans_data= pipeline.transform(features)
    print(trans_data)
    predict_proba = lr.predict_proba(trans_data)
    prediction = lr.predict(trans_data)
    if prediction == 1:
        Pred = "High"
    else:
        Pred = "Low"
    
    predict_proba = ",".join(map(str, np.round(predict_proba[:, 1]*100,2)))
    predict_proba = predict_proba + "%"
    return render_template("result.html", prediction = Pred, predict_proba = predict_proba)

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load(r'C:\Users\ashish.gupta\Downloads\Flask Activity Titanic Project\my_model_titanic.pkl') # Load "model.pkl"
    print ('Model loaded')
    pipeline = joblib.load(r'C:\Users\ashish.gupta\Downloads\Flask Activity Titanic Project\my_pipe_titanic.pkl')
    print ('Pipeline loaded')
  
    app.run(port=port, debug=True)