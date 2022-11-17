import pickle
import keras
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, session, json

app = Flask(__name__)
model = load_model('static/crude_oil.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        x = []
        for i in range(1, 11):
            x += [float(request.form[f'year{i}'])]
        x = np.array(x).reshape(1, -1)

        n_steps = 10
        output = []
        temp_x = list(x)[0].tolist()
        # temp_x = temp_x[0].tolist()

        for i in range(10):
            if(len(temp_x) > 10):
                x = np.array(temp_x[1:])
                x = x.reshape(1, -1)
                x = x.reshape((1, n_steps, 1))
                yhat = model.predict(x, verbose=0)
                temp_x.extend(yhat[0].tolist())
                temp_x = temp_x[1:]
                output.extend(yhat.tolist())
            else:
                x = x.reshape((1, n_steps,1))
                yhat = model.predict(x, verbose = 0)
                temp_x.extend(yhat[0].tolist())
                output.extend(yhat.tolist())

        return render_template('predict.html', predictedResult='The predicted price is:{}'.format(output[0][0]))
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)