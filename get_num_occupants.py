"""This module receives raw data from motes and resolves number of occupants"""

import pickle
import os

from sklearn import linear_model
from scipy import ndimage
import numpy as np

TRAINED_MODEL_FILE = os.path.join(os.getcwd(), "model.pickle")
BACKGROUND_FILE = os.path.join(os.getcwd(), "backgrounds.pickle")

def get_samples(dir_name="dataTrace"):
    """Fetches data from files for use in training"""
    #print "Getting samples"

    dir_name = os.path.join(os.getcwd(), dir_name)
    file_list = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    print("Reading from files:", file_list)

    X = []
    y = []
    for file_name in file_list:
        label = file_name.split("/")[-1].split(".")[0]

        #lines = handle.readlines();
        lines = [[int(i) for i in line.rstrip('\n').split(" ")] for line in open(file_name)]

        # For each file, append each line to X (each line being an 8x8 thermosense grid)
        X.extend(lines)
        y.extend([int(label)] * len(lines))

    return (X, y)

def get_features(active_pixels):
    """Extracts # active pixels, # connected components, and size of largest component"""
    #print "Getting features"

    # print "\n", active_pixels
    struct = np.ones((3, 3), dtype="bool8")
    blobs, number_of_blobs = ndimage.label(active_pixels, structure=struct)

    blob_sizes = []
    for i in range(1, number_of_blobs+1):
        #print len(np.where(blobs == i)[0])
        blob_sizes.append(len(np.where(blobs == i)[0]))

    largest_blob = max(blob_sizes) if blob_sizes else 0

    num_active_pixels = len(np.where(active_pixels == 1)[0])
    # print(active_pixels)
    # print (f"Active Pixels: {num_active_pixels}, Blobs: {len(blob_sizes)}, Largest: {largest_blob}")
    return [num_active_pixels, len(blob_sizes), largest_blob]

def train_model():
    """Trains and saves linear model to be used later"""
    print ("Training model")

    # Grab training data from directory
    (X, y) = get_samples()

    # Train linear model
    feature_list = []
    for x_val in X:
        features = get_features(np.asarray(x_val).reshape(8, 8))
        feature_list.append(features)

    print(f"Feature list: {feature_list}\n")
    print(f"y: {y}")

    print ("Training with {} points".format(len(feature_list)))

    print(len(y))

    ## Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(feature_list, y)

    print (f"Regression Coefficient: {regr.coef_}")
    print (f"Regression Intercept: {regr.intercept_}")

    # Save linear model to TRAINED_MODEL_FILE
    with open(TRAINED_MODEL_FILE, 'wb') as handle:
        pickle.dump(regr, handle, protocol=pickle.HIGHEST_PROTOCOL)

def update_background(grid, pir_is_active, node_id):
    """Uses freshest data to update background"""

    PIR_THRESHOLD = 10
    ALPHA = 0.85

    # if BACKGROUND_FILE exists, load it into background dictionary,
    if os.path.isfile(BACKGROUND_FILE):
        with open(BACKGROUND_FILE, 'rb') as handle:
            background_dict = pickle.load(handle)

    # else make new dictionary for backgrounds
    else:
        background_dict = {}

    # Each node's background entry has its background and time since pir was last active
    if node_id in background_dict.keys():
        (cur_bg, pir_counter) = background_dict[node_id]

    else:
        # Assign initial bg to current grid
        cur_bg = grid
        pir_counter = 0

    # Update pir counter of this node
    if pir_is_active:
        # Reset counter if active
        pir_counter = 0
    else:
        pir_counter += 1

    # Update bg if pir hasn't been active lately
    if pir_counter >= PIR_THRESHOLD:
        # Blend it together
        # print ('Updating Background')
        cur_bg = ((1-ALPHA) * cur_bg + ALPHA*grid)
    else:
        print ('Need %d more idle snaps'%(PIR_THRESHOLD - pir_counter))

    # Update dictionary entry for this node
    background_dict[node_id] = (cur_bg, pir_counter)

    # Save dictionary to BACKGROUND_FILE
    with open(BACKGROUND_FILE, 'wb') as handle:
        pickle.dump(background_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_cur_background(node_id):
    """Queries bg history for node to be used in bg_sub"""
    #print "Getting current background"

    # Reasonable default value
    cur_bg = np.ones((8, 8))*21

    if os.path.isfile(BACKGROUND_FILE):
        with open(BACKGROUND_FILE, 'rb') as handle:
            background_dict = pickle.load(handle)

        if node_id in background_dict.keys():
            # If we have a dictionary AND our node has an entry, we use it
            cur_bg = background_dict[node_id][0]

    # Else use default bg
    return cur_bg

def resolve_num_people(features):
    """Queries linear model with feature list"""

    #print "Resolving number of people"
    features = np.asarray(features).reshape(1, -1)

    # If model file doesn't exist, call for model to be trained
    if not os.path.isfile(TRAINED_MODEL_FILE):
        train_model()

    # Load trained model from pickle
    with open(TRAINED_MODEL_FILE, 'rb') as handle:
        regr = pickle.load(handle)

    # Perform regression on provided features
    return regr.predict(features)[0]


def get_num_occupants(grid, pir_is_active, node_id):
    """Typical program entry to fetch num occupants under sensor"""

    # Sensitivity - pixels deviating more than this amount from average are considered active
    SENS = 3

    # Force grid into 2D shape to be used
    grid = np.asarray(grid).reshape(8, 8)

    # Grab background from history
    background = get_cur_background(node_id)

    # Perform background subtraction and find pixels that exceed our sensitivity threshold
    bg_sub = np.abs(background - grid)
    active_pixels = np.greater(bg_sub, np.ones((8, 8))*SENS)
    
    print(active_pixels)
    # AP = np.asarray(active_pixels)
    # Use active pixels to get features for model query
    features = get_features(active_pixels)

    # Call for background to be updated
    update_background(grid, pir_is_active, node_id)

    # Finally, use trained model to resolve number of people using features
    return (int(round(resolve_num_people(features))))

def get_occupancy_from_active(active_pixels):
    """Debug program entry to test trained model"""

    # Force grid into 2D shape to be used
    active_pixels = np.asarray(active_pixels).reshape(8, 8)
    print (active_pixels)

    # Use active pixels to get features for model query
    features = get_features(active_pixels)
    #print features

    # Finally, use trained model to resolve number of people using features
    return resolve_num_people(features)
