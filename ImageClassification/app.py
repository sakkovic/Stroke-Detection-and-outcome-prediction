from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import keras
import cv2
import tensorflow as tf
import os
import numpy as np
import pickle
import numpy as np


app = Flask(__name__)

dic = {0: 'Normale', 1: 'Stroke'}

model = load_model(
    'C:/Users/MSI GF63/Desktop/ImageClassification/models/imageclassifier.h5')
modelpred = pickle.load(
    open('C:/Users/MSI GF63/Desktop/ImageClassification/models/predstroke.pkl', 'rb'))


model.make_predict_function()


def predict_label(img_path):
    i = cv2.imread(img_path, 0)
    new_size = (256, 256)
    i = cv2.resize(i, new_size)
    p = model.predict(np.expand_dims(i, 0))
    if ((p[0][0]) > 0.5):
        p[0][0] = 1
    else:
        p[0][0] = 0
    return dic[p[0][0]]

# routes


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
    return render_template("index.html", prediction=p, path=img_path)


@app.route('/strokepred.html', methods=['GET', 'POST'])
def strokepred():
    return render_template('strokepred.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data1 = float(request.form['No Chronic Med Condition'])
        data2 = float(request.form['AgeGroup'])
        data3 = float(request.form['Living Situation'])
        data4 = float(request.form['Other Insurance'])
        data5 = float(request.form['Endocrine Condition'])
        data6 = float(request.form['Other Chronic Med Condition'])
        data7 = float(request.form['Obesity'])
        data8 = float(request.form['Special Education Services'])
        data9 = float(request.form['Sex'])
        data10 = float(request.form['Child Health Plus Insurance'])
        data11 = float(request.form['Medicaid Managed Insurance'])
        data12 = float(request.form['Autism Spectrum'])
        data13 = float(request.form['Private Insurance'])
        data14 = float(request.form['Medicare Insurance'])
        data15 = float(request.form['Heart Attack'])
        data16 = float(request.form['Transgender'])
        data17 = float(request.form['Liver Disease'])
        data18 = float(request.form['Drug Substance Disorder'])
        data19 = float(request.form['Criminal Justice Status'])
        arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9,
                       data10, data11, data12, data13, data14, data15, data16, data17, data18, data19]])
        print(arr)
        predstroke = modelpred.predict(arr)
        return render_template('index.html', predictionstroke=predstroke[0])


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
