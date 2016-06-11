#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import datetime

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet",
                    help="Path to Training Set",
                    required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id]*len(class_path)
    class_id += 1

# Create feature extraction and keypoint detector object
print("Creating SIFT keypoints and descriptors: %s" % str(datetime.datetime.now()))
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

for image_path in image_paths:
    im = cv2.imread(image_path)
    kp, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))

print("Stacking descriptors: %s" % str(datetime.datetime.now()))
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
print("Performing k-means clustering: %s" % str(datetime.datetime.now()))
k = 100
voc, variance = kmeans(descriptors, k, 1)

print("Calculating histogram of features: %s" % str(datetime.datetime.now()))
# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

print("Performing Tf-Idf vectorization: %s" % str(datetime.datetime.now()))
# Perform Tf-Idf vectorization
nbr_occurrences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(
    np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurrences + 1)),
    'float32')

# Scaling the words
print("Scaling the words: %s" % str(datetime.datetime.now()))
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
print("Training the Linear regression: %s" % str(datetime.datetime.now()))
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

print("Saving to pkl file: %s" % str(datetime.datetime.now()))
# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)
