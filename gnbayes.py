import csv
import numpy as np 
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score

def read_data():
    # open and read file, store in nuimpy array
    f = open('spambase.data')
    reader = csv.reader(f)
    alldata = np.array(list(reader))

    # seperate labels and data
    data = np.array([x[:-1] for x in alldata]).astype(np.float)
    label = np.array([x[-1] for x in alldata]).astype(np.float)

    # return labels and data for train and test, and split 50/50
    return train_test_split(data, label, test_size=0.5, random_state=RandomState())
    
def prob_model(data, label):
    # classes is 0 or 1, for spam or not
    # class_count finds number of each class in data
    # priors finds probability of each class 
    # nclass is 2, number of classes
    classes, class_counts = np.unique(label, return_counts=True)
    priors = class_counts / label.shape[0]
    nclass = len(classes)
    std = []
    mean = []

    # for each class, 0 or 1 find mean and var of each feature
    for y in classes:
        x = data[label == y, :]
        mean.append(np.mean(x, axis=0))
        std.append(np.std(x, axis=0))

    # vstack stacks arrays vertically
    mean = np.vstack(mean)
    std = np.vstack(std)

    # set any var of 0 to .0001 to avoid divide by 0
    std[std == 0] = 0.0001

    return classes, class_counts, priors, nclass, std, mean

def gnb(data, classes, class_counts, priors, nclass, std, mean):
    # stores list of probability
    prob = []

    # for each class calculate probability
    for k in range(nclass):
        # mean and std are for each class, and has 57 features
        m = mean[k, :]
        s = std[k, :]
        # this is the probability density function
        ex = np.exp(-((data - m)**2 / (2 * s**2)))
        p = 1 / (np.square(2 * np.pi) * s) * ex
        p[p == 0] = 0.0001      # had to add this since it kept giving me divide by zero error
        # sums probabilities
        prob.append(np.sum(np.log(priors[k]) + np.log(p)))

    prob = np.vstack(prob).T
    # finds the argmax which is the prediction 
    return classes[np.argmax(prob, axis=1)]


# read data file and store in arrays
data_train, data_test, label_train, label_test = read_data()

# create probabilistic model  
classes, class_counts, priors, nclass, std, mean = prob_model(data_train, label_train)

# run guassian naive bayes on test data storing results in classnb
classnb =[]
for i in data_test:
    classnb.append(gnb(i, classes, class_counts, priors, nclass, std, mean))

classnb = np.vstack(classnb).T
classnb = classnb.flatten()

#find accuracy of test
accur = accuracy_score(label_test, classnb) * 100
print("Accuracy of test:")
print(accur)
print("Confusion Matrix:")
print(confusion_matrix(label_test, classnb))





