from genericpath import isfile
from get_num_occupants import get_num_occupants
from datetime import datetime, timedelta
import numpy as np
# import pickle
import os
import sys
import tos
import time
import csv
import threading

readings = {}

class gridData(tos.Packet):
    def __init__(self, payload=None):
        tos.Packet.__init__(self,
                            [('data',  'blob', None)],
                            payload)



def logData(snap, pir, node):
    a = np.asarray(snap, dtype=np.float64).reshape((8, 8))/4
    numOccupants = get_num_occupants(a, pir, node)

    currentTime = datetime.now()-timedelta(hours=7)
    readings[node] = numOccupants

    print(f"Pipeline.logData: Num Occupants for Node {node} at {currentTime.strftime('%H:%M:%S')}: {numOccupants}")
    print(a)

    # Create Files for Logging, if not already.
    filename = f"./logs/nodes/{node}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Log Individual Sensors
    with open(filename, 'a') as file:
        file.write(f"{node},{currentTime.strftime('%x')},{currentTime.strftime('%X')},{pir},{numOccupants}\n{snap}\n")

    # print(f"Snap Logged is {snap}")

    sumOccupants = 0
    # Log Sum of All
    for node in readings:
        sumOccupants += readings[node]


def handleSerial(am):
    p = am.read()
    # print(f"\np = {gridData(p.data)}\n")
    if p:
        grid = gridData(p.data)
        
        # From grid, get Node ID, PIR Activity, and Thermal Snap
        node = (grid.data[0] & 0x7f)
        pir = bool(((grid.data[0] & 0x80) != 0))
        snap = grid.data[1:65]

        return (snap, pir, node)

    print("Issue with Serial Handling")


def getOccupants():
    # Main Read Loop
    am = tos.AM()
    while True:
        try:
            (snap, pir, node) = handleSerial(am)
            logData(snap, pir, node)
        except Exception as e:
            print(e)

def printSum():
    threading.Timer(10, printSum).start()
    currentTime = datetime.now()-timedelta(hours=7)
    sumOccupants = 0
    # Log Sum of All
    for node in readings:
        sumOccupants += readings[node]
    print(f"Sum at {currentTime}: {sumOccupants}")
    sum_filename = f"./logs/sum.csv"
    os.makedirs(os.path.dirname(sum_filename), exist_ok=True)
    with open(sum_filename, 'a') as file:
        file.write(f"{currentTime},{sumOccupants}\n")

def main():
    printSum()
    getOccupants()


if __name__ == '__main__':
    main()