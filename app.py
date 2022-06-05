import os
import cv2
from graphviz import render
import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
import src.config as config

# folder location
image_folder = './test_images'

# getting prediction
model = load_model('input/models/' + 
        f'fold_{config.FOLD_IDX}_{config.MODEL_NAME}_bs_{config.BATCH_SIZE}_epochs_{config.N_EPOCHS}_lr_{config.LEARNING_RATE}')

def get_prediction(img_loc, model):
    img = cv2.imread(img_loc)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = np.expand_dims(img, axis=0)

    y_score = model.predict(img).ravel()
    pred_class = np.argmax(y_score)
    prob = y_score[pred_class]
    pred_class = 'Cat' if pred_class == 0 else 'Dog'
    return prob, pred_class

# initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        img_file = request.files['image']

        if img_file:
            img_loc = os.path.join(image_folder, img_file.filename)
            score, pred_class = get_prediction(img_loc, model)
            return render_template('index.html', pred_class=pred_class, score = score, filename = img_file.filename)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=3000)    