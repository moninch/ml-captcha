import argparse
import json
import string
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools

import os
import cv2
import numpy as np
from random import random, randint, choices
import tensorflow as tf
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
import matplotlib.pyplot as plt

alphabet_all = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # QWERTYUIOPLKJHGFDSAZXCVBNM')
num_alphabet = len(alphabet)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Ограничиваем память на GPU (например, до 4GB)
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     except RuntimeError as e:
#         print(e)

def _gen_captcha(img_dir, num_of_letters, num_of_repetition, width, height):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    print("Начинается генерация капчи...")
    image = ImageCaptcha(width=width, height=height)

    for counter in range(num_of_repetition):
        i = choices(alphabet_all, k=5)
        captcha = "".join(i)
        fn = os.path.join(img_dir, "%s_%s.png" % (captcha, uuid.uuid4()))
        image.write(captcha, fn)
        if counter % 1000 == 0:
            print(f"Сгенерировано {counter} изображений из {num_of_repetition}")

    print("Генерация капчи завершена.")


def gen_dataset(path, num_of_repetition, num_of_letters, width, height):
    _gen_captcha(
        os.path.join(path, "data"), num_of_letters, num_of_repetition, width, height
    )
    print("Finished Data Generation")


BATCH_SIZE = 128
NUM_OF_LETTERS = 5
EPOCHS = 50
IMG_ROW, IMG_COLS = 50, 135

# Non-configs
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, "train")


def load_data(path, test_split=0.1):
    print("loading dataset...")
    y_train = []
    y_test = []
    x_train = []
    x_test = []

    # r=root, d=directories, f = files
    counter = 0
    for r, d, f in os.walk(path):
        for fl in f:
            if ".png" in fl:
                flr = fl.split("_")[0]
                counter += 1
                label = np.zeros((NUM_OF_LETTERS, num_alphabet))
                for i in range(NUM_OF_LETTERS):
                    label[i, alphabet.index(flr[i])] = 1
                #                 label = np.zeros((50, 1))
                #                 for i in range(5):
                #                     label[i*5+int(flr[i])] = 1

                img = cv2.imread(os.path.join(r, fl))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (int(IMG_COLS/2), int(IMG_ROW/2)), interpolation=cv2.INTER_AREA)
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))

                if random() < test_split:
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)

    print("Размер набора данных:", counter)
    print(
        f"Обучающая выборка: {len(y_train)} изображений, Тестовая выборка: {len(y_test)} изображений"
    )
    print("dataset size:", counter, "(train=%d, test=%d)" % (len(y_train), len(y_test)))
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


if not os.path.exists(DATA_PATH):
    print("Generating Dataset")
    gen_dataset(DATA_PATH, 300 * 1000, NUM_OF_LETTERS, IMG_COLS, IMG_ROW)

# Generating Dataset
# Finished Data Generation


x_train, y_train, x_test, y_test = load_data(DATA_PATH)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255

# loading dataset...
# dataset size: 10000 (train=89mod53, test=1047)

s_train = []
s_test = []
for i in range(NUM_OF_LETTERS):
    s_train.append(y_train[:, i, :])
    s_test.append(y_test[:, i, :])

save_dir = os.path.join(PATH, "saved_models")
model_name = "keras_cifar10_trained_model.h5"

input_layer = Input((int(IMG_ROW/2), int(IMG_COLS/2), 1))
x = Conv2D(filters=32, kernel_size=(5, 5), padding="same", activation="relu")(
    input_layer
)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=48, kernel_size=(5, 5), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Dropout(0.3)(x)
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)

out = [
    Dense(num_alphabet, name="digit%d" % i, activation="softmax")(x)
    for i in range(NUM_OF_LETTERS)
]
# out = Dense(num_alphabet*5, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=out)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[["accuracy"] for _ in range(NUM_OF_LETTERS)])


model.summary()

hist_train_loss_digit = {i: [] for i in range(5)}
hist_test_loss_digit = {i: [] for i in range(5)}

hist_train_acc_digit = {i: [] for i in range(5)}
hist_test_acc_digit = {i: [] for i in range(5)}

hist_train_loss = []
hist_test_loss = []

hist_train_acc = []
hist_test_acc = []

digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
val_digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
loss = []
val_loss = []

# Обучение модели с добавлением информации об эпохах
history = model.fit(
    x_train,
    s_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(x_test, s_test),
)
# -------------------

digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
val_digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
loss = []
val_loss = []


def plot_diagram(digit_acc_now, val_digit_acc_now, loss_now, val_loss_now):
    global digit_acc, val_digit_acc, loss, val_loss

    for i in range(NUM_OF_LETTERS):
        digit_acc[i].extend(digit_acc_now[i])
        val_digit_acc[i].extend(val_digit_acc_now[i])
    loss.extend(loss_now)
    val_loss.extend(val_loss_now)

    for i in range(NUM_OF_LETTERS):
        s = {0: "First", 1: "Second", 2: "Third", 3: "Fourth", 4: "Fifth"}[i]
        # plt.plot(val_digit_acc[i], label='%s Digit Train' % s)
        plt.plot(digit_acc[i], label="%s Digit Test" % s)

    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    for i in range(NUM_OF_LETTERS):
        s = {0: "First", 1: "Second", 2: "Third", 3: "Fourth", 4: "Fifth"}[i]
        plt.plot(val_digit_acc[i], label="%s Digit Train" % s)
        # plt.plot(digit_acc[i], label='%s Digit Test' % s)

    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    # Plot training & validation loss values

    plt.plot(val_loss, label="Train")
    plt.plot(loss, label="Test")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


plot_diagram(
    [history.history["digit%d_accuracy" % i] for i in range(NUM_OF_LETTERS)],
    [history.history["val_digit%d_accuracy" % i] for i in range(NUM_OF_LETTERS)],
    history.history["loss"],
    history.history["val_loss"],
)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("Saved trained model at %s " % model_path)

# Score trained model.
scores = model.evaluate(x_train, s_train, verbose=1)
print("Train loss:     %f" % np.mean(scores[0:5]))
acc = 1.0
for i in range(5):
    acc *= scores[6 + i]
print("Train accuracy: %.2f" % (acc * 100.0))

scores = model.evaluate(x_test, s_test, verbose=1)
print("Test loss:     %f" % np.mean(scores[0:5]))
acc = 1.0
for i in range(5):
    acc *= scores[6 + i]
print("Test accuracy: %.2f" % (acc * 100.0))