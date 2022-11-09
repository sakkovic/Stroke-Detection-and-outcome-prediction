from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import keras
import cv2
import tensorflow as tf
import os
import numpy as np


app = Flask(__name__)

dic = {0: 'Normale', 1: 'Stroke'}

model = load_model(
    'C:/Users/MSI GF63/Desktop/ImageClassification/models/imageclassifier.h5')

model.make_predict_function()

# def predict_label(img_path):
# # img = cv2.imread(img_path,0)
# # img = cv2.resize(img, (256,256))
# # p = model.predict(np.expand_dims(img,0))
# # i = image.load_img(img_path, target_size=((256,256)))
# i = image.img_to_array(i)/255.0
# i = i.reshape(1, 256,256,1
# p = model.predict_classes(i)
# return dic[p[0]]


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
        # img_path = "C:\\Users\\MSI GF63\\Desktop\\ImageClassification\\static\\"
        # img_path = img_path + img.filename
        # img_path = img.filename
        print('img_path: ', img_path)

    return render_template("index.html", prediction=p, path=img_path)


if __name__ == '__main__':
    #app.debug = True
    app.run(debug=True)
