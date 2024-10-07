from sklearn.model_selection import train_test_split
from sklearn import linear_model
from scipy import ndimage
import numpy as np
import random
import sys
import os

def getFeatures(X):
	# Get # active pixels, # connected components, and size of largest component
	features = [];

	for x in X:
		im = np.reshape(np.asarray(x), (8, 8))
		# print "\n", im
		struct=np.ones((3,3), dtype="bool8")
		blobs, number_of_blobs = ndimage.label(im, structure = struct)

		blobSizes = [];
		for i in range(1,number_of_blobs+1):
			#print len(np.where(blobs == i)[0])
			blobSizes.append(len(np.where(blobs == i)[0]))

		largestBlob = max(blobSizes) if len(blobSizes)>0 else 0;

		activePixels = len(np.where(im == 1)[0])

		#print "Active: {}, # blobs {}, largest {}".format(activePixels, len(blobSizes), largestBlob)
		features.append([activePixels, len(blobSizes), largestBlob])

	return features;

def getSamples(dirName):
	dirName = os.path.join(os.getcwd(),dirName);
	fileList = [os.path.join(dirName, f) for f in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, f))]
	print "Reading from files:", fileList

	X = []; y = [];
	for fileName in fileList:
		label = fileName.split("/")[-1].split(".")[0];

		#lines = handle.readlines();
		lines = [[int(i) for i in line.rstrip('\n').split(" ")] for line in open(fileName)]


		X.extend(lines);
		y.extend([int(label)] * len(lines))

	return (X,y)

'''
	If we're pulling testing and training from the same dataset, we must split it
'''
def readAndSplit(dirName, **kwargs):
	split = kwargs["split"] if "split" in kwargs.keys() else .1;
	(X,y) = getSamples(dirName)
	x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=split);
	return (x_train, x_test, y_train, y_test)

def read(trainDir, testDir):
	(x_train, y_train) = getSamples(trainDir);
	(x_test, y_test) = getSamples(testDir);
	return (x_train, x_test, y_train, y_test)

if len(sys.argv) < 3:
	print "Usage: {} trainingDir testingDir".format(sys.argv[0]);
	exit(1)

if sys.argv[1] == sys.argv[2]:
	print "Testing and training use the same dataset!";
	(x_train, x_test, y_train, y_test) = readAndSplit(sys.argv[1], split=.3);

else:
	(x_train, x_test, y_train, y_test) = read(sys.argv[1], sys.argv[2]);


### Now we fetch our features for regression ###

test_features = getFeatures(x_test)
train_features = getFeatures(x_train)

print "{} training, {} testing".format(len(x_train), len(x_test));

## Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(train_features, y_train)

print regr.coef_
print regr.intercept_

pred = regr.predict(test_features)
rounded = [int(round(x)) for x in pred];

for i in range(0,len(y_test)):
	print "Y: {}\tYpred: {}".format(y_test[i], int(round(pred[i])))
	if y_test[i] != int(round(pred[i])):
		print np.reshape(np.asarray(x_test[i]), (8, 8))
		print test_features[i]
		print ""

diffs = [i for (i,_) in enumerate(rounded) if rounded[i] != y_test[i]]

print "{}/{} misclassified, {}%".format(len(diffs), len(pred), (100.0*len(diffs))/len(pred))

#print "W is of size {}x{}".format(len(regr.coef_), len(regr.coef_[0]))
#print "w is of size {}x{}".format(len(regr.intercept_), 1)
print regr.coef_
print regr.intercept_

print("Mean squared error: %.2f" % np.mean((regr.predict(test_features) - y_test) ** 2))
print('Variance score: %.2f' % regr.score(test_features, y_test))
