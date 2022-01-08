from flask import Flask, render_template, request, send_from_directory
from zipfile import ZipFile
import cv2
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

ZipFile("static/model.zip").extractall("static/")

model = Sequential()

model.add(InputLayer(input_shape=[150,150,3]))

model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, padding='same'))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['acc'])

model.load_weights('static/model.h5')


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/hasilprediksi', methods=['POST'])
def home():
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (150,150))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 150,150,3)
    prediction = model.predict(img_arr)

    pre = round(prediction[0,0], 1)
    preds = np.array([pre])
    COUNT += 1
    return render_template('prediksi.html', data=preds)


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)



