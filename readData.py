from re import A
from get_num_occupants import get_num_occupants
from datetime import datetime, timedelta
import numpy as np
import os
import sys
import tos
import time
import csv

# row[0]: Node ID, or Grid Data (RAW)
# row[1]: Date
# row[2]: Time
# row[3]: PIR
# row[4]: Occupancy

dictionary = {}

def getOccupants(path):
    if(not os.path.exists(path)):
        print(f"File '{path}' does not exist!\n")
        sys.exit()
    with open(path, newline='') as file:
        reader = csv.reader(file)
        i = 0

        # Initialize variables we'll get from the CSV log file
        node = 0
        pir = 0
        date = 0
        time = 0

        for row in reader:
            # Rows 1,3,5, etc., contain our main data
            if(i % 2 == 0):
                node = row[0]
                date = row[1]
                time = row[2]
                pir  = row[3]

            # Rows 2,4,6, etc., contain our thermo array data
            # Every other row we are able to get combine these together
            elif(i % 2 != 0):
                # Clean up log file for use
                for j, s in enumerate(row):
                    row[j] = s.replace('[', '').replace(']', '')

                # Just like pipeline.py, construct array from thermo data
                # and find num_occupants from get_num_occupants.py
                data = [int(x) for x in row]
                snap = np.asarray(data, dtype=np.float64).reshape((8, 8))/4
                occupants = get_num_occupants(snap, pir, node)
                
                # DEBUG
                print(f"occupants for node {node} at {time}: {occupants}")
                print(snap)

                # Create Files for Logging, if not already.
                filename = f"./offline_logs/nodes/{node}.csv"
                # if os.path.exists(filename): os.remove(filename) # Clear it
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                sum_filename = f"./offline_logs/sum.csv"
                # if os.path.exists(filename): os.remove(sum_filename) # Clear it
                os.makedirs(os.path.dirname(sum_filename), exist_ok=True)

                # Individual Files: Should equal the live log, minus thermal data.
                with open(filename, 'a') as file:
                    file.write(f"{node},{date},{time},{occupants}\n")

                # Summation: Adds dates and times to dictionary.
                date2 = datetime.strptime(date, '%x')
                time2 = datetime.strptime(time, '%X').time()

                dt = datetime.combine(date2, time2)
                dictionary[dt] = occupants

            # Increment line in live log
            i = i + 1

def printDictionary():
    # Sort Dictionary
    sorted_dict = {k: dictionary[k] for k in sorted(dictionary)}
    
    # Get first Datetime value (DEBUG: and last value)
    firsttime = list(sorted_dict.keys())[0]
    lasttime = list(sorted_dict.keys())[-1]

    # Set the "nexttime" to be 10 seconds after our first time
    nexttime = firsttime + timedelta(seconds=10)
    currentsum = 0
    sum = 0

    # DEBUG
    print(f"Dictionary: {firsttime} -> {lasttime}")

    with open ("./offline_logs/sum.csv", 'w') as file:
        for key, value in sorted_dict.items():
            if (key < nexttime):
                currentsum += sorted_dict[key]
            else:
                file.write(f"{nexttime}, {currentsum}\n")
                while(key > nexttime):
                    nexttime += timedelta(seconds=10)
                currentsum = 0
            
    # file.write(f"{key}, {value}\n")

def getDirectories():
    dir = './logs/nodes/'
    for filename in os.listdir(dir):
        # Get all individual logs.
        f = os.path.join(dir, filename)
        getOccupants(f)
    # getOccupants('./logs/nodes/7.csv')
    printDictionary()


def main():
    getDirectories()

if __name__ == '__main__':
    main()