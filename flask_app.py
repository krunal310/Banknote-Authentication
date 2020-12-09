from flask import Flask, request
import pandas as pd
import numpy as np
import tensorflow as tf

app=Flask(__name__)
model = tf.keras.models.load_model('E:\PROJECT\Banknote-Authentication\model.h5')


@app.route('/')
def welcome():
    return "Welcome to Bank Note Authentication App"

if __name__=='__main__':
    app.run()