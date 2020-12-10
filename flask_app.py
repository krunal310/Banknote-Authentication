from flask import Flask, request
import pandas as pd
import numpy as np
import tensorflow as tf

app=Flask(__name__)
model = tf.keras.models.load_model('E:\PROJECT\Banknote-Authentication\model.h5')


@app.route('/')
def welcome():
    return "Welcome to Bank Note Authentication App"

@app.route('/predict')
def predict():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=np.argmax(model.predict([[float(variance),float(skewness),float(curtosis),float(entropy)]]), axis=-1) 
    return "Predicted Class : "+ str(prediction[0])
if __name__=='__main__':
    app.run()