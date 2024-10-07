from ctypes import sizeof
from genericpath import isfile
from tkinter import Y
from get_num_occupants import get_num_occupants
from datetime import datetime, timedelta
import numpy as np
import pickle

# import pickle
import os
import sys
import tos
import time
import csv
import threading

import tensorflow as tf
from tensorflow import keras
print("TensorFlow version:", tf.__version__)

thermal_data_set = [0]*10
predicted_from_getnum = []
snap = 0

def get_samples(dir_name="dataTrace"):
    """Fetches data from files for use in training"""

    # Gets directory name of "datatrace" folder, if exists, then creates a list of files within that folder.
    dir_name = os.path.join(os.getcwd(), dir_name)
    file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]    

    X = [] # Lines
    y = [] # Num Samples

    
    for file_name in file_list:
        label = file_name.split("/")[-1].split(".")[0]
        lines = [[int(i) for i in line.rstrip('\n').split(" ")] for line in open(file_name)]
        X.extend(lines)
        # for line in lines:
        #     X.extend(line)
        #     break
        
        for j in range(len(lines)):
            yi = [0,0,0,0]
            yi[int(label)] = 1
            y.append(yi)
        # y.extend([int(label)] * len(lines))
        # break
    
    
    # X = np.asarray(X, dtype=np.float64).reshape((8,8))
    X,y = np.array(X), np.array(y)
    print(len((lines)))
    print(X.shape, y.shape)
    return X,y


# Mote Data Handling
class gridData(tos.Packet):
    def __init__(self, payload=None):
        tos.Packet.__init__(self,
                            [('data',  'blob', None)],
                            payload)

def handleSerial(am):
    p = am.read()
    if p:
        grid = gridData(p.data)
        
        # From grid, get Node ID, PIR Activity, and Thermal Snap
        node = (grid.data[0] & 0x7f)
        pir = bool(((grid.data[0] & 0x80) != 0))
        snap = grid.data[1:65]
        return (snap, pir, node)

    print("Issue with Serial Handling")

def init_network():
    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(8,)))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))


    model = tf.keras.models.Sequential()
    # model.add(tf.keras.Input(shape=(8,)))
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8)))
    model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(80, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))
    
    return (model)

def network(model, thermal_node_data):
    X,y = get_samples()
    predictions = model(X)
    print(predictions)
    model.summary()

    tf.nn.softmax(predictions).numpy()
    # truth = get_samples()
    # truth = np.array([0,1,0,0]).reshape((1,4))
    # truth = 0
    # Loss Function

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss_fn(y, predictions).numpy()

    # Backward Propagation

    print("Debug: Checkpoint 0")

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])
    
    print("Debug: Checkpoint 1")

    model.fit(predictions, y, epochs=5)

    print("Debug: Checkpoint 2")

    # saving the trained model
    model.save("nn_model")



# Main Loop
def getOccupants():
  am = tos.AM()
  model = init_network()
  while True:
      try:
          (snap, pir, node) = handleSerial(am)
        #   print(len(snap))
          thermal_node_data = np.asarray(snap, dtype=np.float64).reshape((1,  64))/4  
        #   thermal_node_data = np.asarray(snap, dtype=np.float64) / 4  
        #   thermal_node_data = np.asarray(snap, dtype=np.float64).reshape((64, 1))/4  
        #   print(thermal_node_data)        

          print(f"Debug: Received input from node {node}")

          if(thermal_data_set[node-1] == 0):
            thermal_data_set[node-1] = thermal_node_data

          network(model, thermal_node_data)
          

      except Exception as e:
          print(e)

def main():
    # get_samples()
    getOccupants()

if __name__ == '__main__':
    main()
