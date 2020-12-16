from flask import Flask, request
import pandas as pd
import numpy as np
import tensorflow as tf
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

model = tf.keras.models.load_model('E:\PROJECT\Banknote-Authentication\model.h5')


@app.route('/')
def welcome():
    return "Welcome to Bank Note Authentication App"


@app.route('/predict', methods=["Get"])
def predict():

    """Banknote Authentication 
    Features are extracted with the use of Wavelet Transform. 
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: kurtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    kurtosis=request.args.get('kurtosis')
    entropy=request.args.get('entropy')
    prediction=np.argmax(model.predict([[float(variance),float(skewness),float(kurtosis),float(entropy)]]), axis=-1) 
    return "Predicted Class : "+ str(prediction[0])

if __name__=='__main__':
    app.run()