import numpy as np
import sklearn
from random import shuffle
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

def main():
    """Builds Convolutional Neural Network for use with pipeline.py"""

    train_labels = []
    train_samples = []
    test_labels = []
    test_samples = []
    
    dir_name = os.path.join(os.getcwd(), "train")
    file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    for file in file_list:
        num_people = file.split("/")[-1].split(".")[0]
        for line in open(file):
            train_labels.append(int(num_people))
            train_samples.append(np.asarray(line.rstrip('\n').split(" ")))

    # dir_name = os.path.join(os.getcwd(), "test")
    # file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    # for file in file_list:
    #     num_people = file.split("/")[-1].split(".")[0]
    #     for line in open(file):
    #         test_labels.append(int(num_people))
    #         test_samples.append(np.asarray(line.rstrip('\n').split(" ")))

    train_labels = np.array(train_labels).reshape(1063, -1)
    train_samples = np.array(train_samples).reshape(1063, 8, 8).astype(np.float)
    # test_labels = np.array(test_labels).reshape(80, -1)
    # test_samples = np.array(test_samples).reshape(80, 8, 8).astype(np.float)

    train_labels, train_samples = shuffle(train_labels, train_samples, random_state=0)
    test_labels, test_samples = shuffle(test_labels, test_samples, random_state=0)

    for i in range(5):
        print(train_labels[i])
        print(train_samples[i])

    model = tf.keras.models.Sequential([
        Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',input_shape=(8,8,1)),
        MaxPool2D(pool_size=(2,2),strides=2),
        Conv2D(filters=64, kernel_size=(3,3),activation='relu', padding='same'),
        MaxPool2D(pool_size=(2,2), strides=2),
        Flatten(),
        Dense(units=4, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("checkpoint1")
    model.fit(x=train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)
    print("checkpoint2")
    model.save()

    # predictions = model.predict(x=test_samples, batch_size=10, verbose=0)
    # rounded_predictions = np.argmax(predictions, axis=-1)

    # print(rounded_predictions)

if __name__ == '__main__':
   main()