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
from tensorflow.keras import backend as K

def main():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    with tf.device('/CPU:0'):
        model = keras.models.load_model('model.h5')

        test_labels = []
        test_samples = []

        print("1")

        dir_name = os.path.join(os.getcwd(), "test")
        file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
        for file in file_list:
            num_people = file.split("/")[-1].split(".")[0]
            for line in open(file):
                test_labels.append(int(num_people))
                test_samples.append(np.asarray(line.rstrip('\n').split(" ")))

        print("2")

        test_labels = np.array(test_labels).reshape(80, -1)
        test_samples = np.array(test_samples).reshape(80, 8, 8).astype(np.float)

        print("3")
        predictions = model.predict(x=test_samples, batch_size=10, verbose=0)
        print("4")
        rounded_predictions = np.argmax(predictions, axis=-1)

        print(rounded_predictions)

    

if __name__ == '__main__':
   main()