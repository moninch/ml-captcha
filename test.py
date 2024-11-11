import cv2
import numpy as np
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
from keras._tf_keras.keras.models import load_model

model = load_model('keras_cifar10_trained_model.h5')
IMG_ROW, IMG_COLS = 50, 135
img = cv2.imread('1234.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (int(IMG_COLS/2), int(IMG_ROW/2)), interpolation=cv2.INTER_AREA)
img = np.reshape(img, (img.shape[0], img.shape[1], 1))

x = img_to_array(img)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
word = ""
for pred in preds:
    word += alphabet[np.argmax(pred)]
print(word)


