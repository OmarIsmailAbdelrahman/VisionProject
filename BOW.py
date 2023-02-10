import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import pickle

import sys

np.set_printoptions(threshold=sys.maxsize)

# Global
#   Variables
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
n_words = 80

# Models
sift = cv2.xfeatures2d.SIFT_create()
kmeans = KMeans(n_clusters=n_words, random_state=0)

# sets value
# set training for training the sift or load model
# uncomment practical test for testing files, and comment the above code
# change the directory for brain and brest in HOG to train the models
# uncomment the main0 to train the svm model


def initialization():
    imgs_des = []  # descriptors of all img
    img_count = 0  # number of images
    label = []  # label of each image
    img_arr = []  # images itself
    return imgs_des, img_count, label, img_arr

def readfile(path):
    imgs_des, count, label, img_arr = initialization()
    type = 0
    img_names = []
    for files in os.listdir(path):
        folder = path + '/' + files
        print(folder)  # folder that is in to it
        for img_file in os.listdir(folder):
            imgfolder = folder + "/" + img_file  # calculate directory of each img
            # print(i,imgfolder)                              # each img directory
            img = cv2.imread(imgfolder, 0)  # read img
            img_names.append(img_file)
            img_arr.append(img)
            kpt, des = sift.detectAndCompute(img, None)  # compute sift
            imgs_des.append((des))
            label.append(type)  # label it
            count += 1
        type += 1
    return imgs_des, label, count, img_arr, img_names

def stacking(des):
    stack = np.array(des[0])
    for i in range(len(des)):
        if i == 0:
            continue
        stack = np.vstack((stack, des[i]))
    return stack

def createBOW(imgs_des, words, img_count, n_words):
    bag = np.zeros((img_count, n_words))
    curr = 0
    img_number = 0
    for i in imgs_des:
        for j in range(i.shape[0]):
            bag[img_number, words[curr + j]] += 1
        curr += i.shape[0]
        img_number += 1
    return bag

def save_model(model, filename):
    # # model saving
    pickle.dump(model, open(filename, 'wb'))

def classification_predict(img_type, X):
    if img_type == "brain":
        filename = "svm for brain.sav"
    else:
        filename = "svm for breast.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(X)
    print(result)

def classification_score(img_type, X, y):
    if img_type == "brain":
        filename = "svm for brain.sav"
    else:
        filename = "svm for breast.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X, y)
    print(result)

def load_model(filename):
    print("reading file ", filename)
    model = pickle.load(open(filename, 'rb'))
    return model


# steps:
# 1. read the images of heart and breast and preprocessing ?
# 2. extract the features of each image and save it in an array
# 3. create a cluster of features and make where each cluster center is visual word
# 4. map every histogram into 1 cluster amd make a histogram of each img containing the histogram of visual words
# 5. normalize the histogram
# 5. train a classifier to classify the histogram into the 2 classes

# 1. read and preprocessing & 2. extract features of each image
training = False            # set true for training and false for loading models ::::::::::::::

if training:  # if need training set to true

    path = os.getcwd()
    print("cur", path)

    imgs_des, label, img_count, img_arr, img_names = readfile(path + "\images")
    print(label)
    # stack all input to train in the k means
    print("number of imgs in train ", len(imgs_des))
    stack = stacking(imgs_des)

    # train the k-means algorithm for n clusters
    words = kmeans.fit_predict(stack)
    save_model(kmeans, "kmean.sav")

    # create the BOW for each img
    bag = createBOW(imgs_des, words, img_count, n_words)

    print(bag.shape)
    # SVM
    clf = SVC()
    clf.fit(bag, label)
    score_value = clf.score(bag, label)  # calculate score
    print("train sift score",score_value)
    save_model(clf, "SVM sift.sav")  # model save

    print("SVM fit")
else:
    kmeans = load_model("kmean.sav")
    clf = load_model("SVM sift.sav")

def sift_predict(dir):
    print("predict")
    path = os.getcwd()

    imgs_des, label, img_count, img_arr, img_names = readfile(path + dir)
    print("number of imgs in test ", len(imgs_des))
    stack = stacking(imgs_des)

    words = kmeans.predict(stack)

    bag = createBOW(imgs_des, words, img_count, n_words)

    classes = clf.predict(bag)
    return classes, label, img_count, img_arr, img_names

# to calculate the test score
def test_score(dir):
    print("test start")
    classes, label, img_count, img_arr, img_names = sift_predict(dir)
    false = 0
    for i in range(classes.shape[0]):
        # print(i, classes[i], label[i])
        if classes[i] != label[i]:
            false += 1
    print("test sift score :", (1 - (false / classes.shape[0])))

#practical functions

def practicalread():
    imgs_des, count, label, img_arr = initialization()
    img_names = []
    path = "C:\\Users\\Legendary\\PycharmProjects\\VisionProject\\practical test\\MultiCancerTestSamples"
    for img_file in os.listdir(path):
        imgfolder = path + "/" + img_file  # calculate directory of each img
        img = cv2.imread(imgfolder, 0)  # read img
        img_names.append(img_file)
        img_arr.append(img)
        kpt, des = sift.detectAndCompute(img, None)  # compute sift
        imgs_des.append((des))
        count += 1
    return imgs_des, count, img_arr, img_names

def sift_predict_practical():
    imgs_des, img_count, img_arr, img_names = practicalread()
    print("number of imgs in test ", len(imgs_des))
    stack = stacking(imgs_des)

    words = kmeans.predict(stack)

    bag = createBOW(imgs_des, words, img_count, n_words)

    classes = clf.predict(bag)
    return classes, img_count, img_arr, img_names


# test_score("\itest")          # test for sift algo
# classes, label, img_count, img_arr, img_names = sift_predict("\itest")
#practical test                             uncomment here and comment the above line
classes, img_count, img_arr, img_names = sift_predict_practical()


# prepare for HOG algorithm
brain = []
brain_name = []
breast = []
breast_name = []
# set two sets for the brain and breast and predict for each one of them
for i in range(img_count):
    if classes[i] == 0:
        # print("brain")
        brain.append(img_arr[i])
        brain_name.append(img_names[i])

    else:
        # print("breast")
        breast.append(img_arr[i])
        breast_name.append(img_names[i])

#############################################################################################################################################

#                   HOG
# split brain and breast to two models
def readfile1(path):
    imgs_des = []
    label = []
    type = 0
    count = 0
    for files in os.listdir(path):
        i = 0
        folder = path + '/' + files
        print(folder)  # folder that is in to it
        for img_file in os.listdir(folder):
            imgfolder = folder + "/" + img_file  # calculate directory of each img
            # print(i,imgfolder)    # each img directory
            img = cv2.imread(imgfolder, 0)  # read img
            img = cv2.resize(img, (64, 128))
            HOG_features = hog(img)
            imgs_des.append((HOG_features))
            label.append(type)  # label it
            count += 1
            i += 1
        type += 1
    return imgs_des, label


# send the img array to prepare for classification of tomur or not
def classify(img_arr):
    imgs_des = []
    for img in img_arr:
        img = cv2.resize(img, (64, 128))
        HOG_features = hog(img)
        imgs_des.append((HOG_features))
    return imgs_des

def main1():    #breast read filename
    # training
    des, label = readfile1("C:\\Users\\Legendary\\PycharmProjects\\VisionProject\\img files\\brain\\train")
    svm = SVC(kernel="linear", C=1)
    print(label)
    svm.fit(des, label)
    acc = svm.score(des, label) * 100
    print("train", acc)

    # testing
    des1, label1 = readfile1("C:\\Users\\Legendary\\PycharmProjects\\VisionProject\\img files\\brain\\test")
    acc = svm.score(des1, label1) * 100
    print("phase 2 test", acc)

    # model saving
    filename = 'svm for brain.sav'
    pickle.dump(svm, open(filename, 'wb'))


def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # dx
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)  # dy
    mag, ang = cv2.cartToPolar(gx, gy)  # magnitude and angel
    bin_n = 16  # Number of bins
    bin = np.int32(bin_n * ang / (2 * np.pi))  #

    bin_cells = []
    mag_cells = []

    cellx = celly = 8

    # 16 rows
    for i in range(0, int(img.shape[0] / celly)):
        for j in range(0, int(img.shape[1] / cellx)):  # 8 columns
            bin_cells.append(bin[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])
            mag_cells.append(mag[i * celly: i * celly + celly, j * cellx: j * cellx + cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    return hist


# call the the HOD from here and change the dir from above
# main1()

#############################################################################################################################################

brain_des = classify(brain)
breast_des = classify(breast)

model_brain = load_model("svm for brain.sav")
model_breast = load_model("svm for breast.sav")

x = model_brain.predict(brain_des)          # tomur - nontomur
y = model_breast.predict(breast_des)        # normal . etc

# print("brain is",x)
# print("breast is",y)

print(len(brain), len(breast))
print("brain prediction")
for i in range(len(brain)):
    if x[i] == 0:
        name = "no tomur"
    else:
        name = "tomur"
    tmp1 = img = cv2.resize(brain[i], (400, 400))            #resize for display
    print(brain_name[i], x[i])
    name = "brain " + name
    cv2.imshow(name, tmp1)
    cv2.waitKey(0)

print("breast prediction")
for i in range(len(breast)):
    if y[i] == 0:
        name = "benign"
    elif y[i] == 1:
        name = "malignant"
    else:
        name = "normal"
    name = "breast " + name
    print(breast_name[i], y[i])
    tmp = img = cv2.resize(breast[i], (400, 400))            #resize for display
    cv2.imshow(name, tmp)
    cv2.waitKey(0)
