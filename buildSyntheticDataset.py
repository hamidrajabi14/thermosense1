import numpy as np
import random
import os

SINGLE_PERSON_TRACE = "dataTrace/1.txt";

readTuples = [];

def randomlyPlace(image, chosenShape):
	#print chosenShape
	# Randomly rotate shape
	chosenShape = np.rot90(chosenShape,random.randint(0,3));

	# Get sizes
	IS = image.shape;
	SS = chosenShape.shape;

	# Choose random location to place it, within image boundaries
	x = random.randint(0,IS[1]-SS[1]);
	y = random.randint(0,IS[0]-SS[0]);

	# Slice shape into image
	image[y:y+SS[0],x:x+SS[1]] = chosenShape;

	return image


def generateImages(uniqueShapes, numPeople, numSamples):
	samples = [];

	for sampleID in range(0,numSamples):
		# print "Generating sample", sampleID;

		# Start with empty image
		image = np.zeros((8,8),dtype=np.int)

		for personID in range(0,numPeople):
			# Pick random shape from uniqueShapes
			chosenShape = random.choice(uniqueShapes);

			# Randomly place shape in image, return it
			image = randomlyPlace(image, chosenShape);

		#print image
		samples.append(image.flatten().tolist());

	return samples;

def getUniqueShapes(lines):
	uniqueShapes = [];

	for item in lines:
		# Grab and reshape this image
		x = np.array([int(i) for i in item.replace("\n","").split(" ")]);
		rs = x.reshape(8,8);

		# Delete empty rows:
		rows, = np.where(np.sum(rs,axis=1)==0)
		rs = np.delete(rs,rows,0)

		# Delete empty cols:
		cols, = np.where(np.sum(rs,axis=0)==0)
		rs = np.delete(rs,cols,1)

		#print rs

		#print shapes
		found = False;
		for shape in uniqueShapes:
			if np.array_equal(rs, shape):
				# Jump to next iteration
				found = True;

		# it was never seen, so append it
		if not found:
			uniqueShapes.append(rs)

	return uniqueShapes

# Parse single-person trace
with open(SINGLE_PERSON_TRACE, 'rb') as handle:
	lines = handle.readlines();

# Get unique shapes from trace
uniqueShapes = getUniqueShapes(lines);

for shape in uniqueShapes:
	print ""
	print shape

print "Found {} unique shapes".format(len(uniqueShapes));

# Make directory to store synthetic data
dataDirectory = os.path.join(os.getcwd(), "synthetic")
if not os.path.exists(dataDirectory):
	os.makedirs(dataDirectory)

numSamples = 10000;
for numPeople in range(0,9):
	print ""
	print "Generating {} samples for {} people".format(numSamples, numPeople);
	images = generateImages(uniqueShapes, numPeople, numSamples);

	traceFile = os.path.join(dataDirectory, "{}.txt".format(numPeople))
	with open(traceFile, "w") as myFile:
		for image in images:
			myFile.write(" ".join([str(s) for s in image])+"\n")
			# print image
