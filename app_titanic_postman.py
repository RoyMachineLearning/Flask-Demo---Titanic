# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys
# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            print('mayy')
            json_ = request.json
            print(',,,,',json_)
            print(json_)
            query = pd.DataFrame(json_)
            print(query)
            trans_data= pipeline.transform(query)
            print(trans_data)
            scaled_df = scalar.transform(trans_data)
            # return to data frame
            query = pd.DataFrame(scaled_df)
            print(query)
            prediction = list(lr.predict(query))
            print({'prediction': str(prediction)})
            return jsonify({'prediction': str(prediction)})
            return "Welcome to titanic model APIs!"

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('C:/courses_centennial/COMP 247/Titanic_project/my_model_titanic.pkl') # Load "model.pkl"
    print ('Model loaded')
    pipeline = joblib.load('C:/courses_centennial/COMP 247/Titanic_project/my_pipe_titanic.pkl')
    print ('Pipeline loaded')
    scalar = joblib.load('C:/courses_centennial/COMP 247/Titanic_project/my_scalar_titanic.pkl')
    print ('scalar loaded')
    
    
    app.run(port=port, debug=True)
