import pandas as pd
import numpy as np
from sklearn import preprocessing


# Build the transformation methods of null values
def add_scaled(column):
    column_new = pd.Series(preprocessing.scale(column))
    column_new.name = column.name + '_scaled'
    return column_new

def add_normalized(column):
    column_new = pd.Series(preprocessing.normalize(column.values.reshape(1, -1), norm='l2').reshape(-1, 1)[:,0])
    column_new.name = column.name + '_normalized'
    return column_new


def replace_null_with_mean(column):
    column_new = pd.Series(column.replace(np.NaN, np.nanmean(column)))
    column_new.name = column.name + '_mean'
    return column_new

def replace_null_with_median(column):
    column_new = pd.Series(column.replace(np.NaN, np.nanmedian(column)))
    column_new.name = column.name + '_median'
    return column_new

# replace with knn
import math
import sklearn
from sklearn.cross_validation import train_test_split
import operator

def euclideanDistance(instance1, instance2):
    length = len(instance1)
    # you can also check if instance1 and instance2 have the same length
    distance = 0
    for l in range(length):
        distance += (instance1[l] - instance2[l])**2
    return math.sqrt(distance)

def getNeighbors(data, labels, testInstance, K):
    distances = []
    neighbors = {}
    #Finds the distances between all the points and creates a list of tuples.
    for i in range(len(data)):
        dist = euclideanDistance(testInstance, data[i, :])
        distances.append([data[i,:], dist])
    idx = np.argsort(np.array(distances)[:, 1]) # get the index in order
    neighbors_data = data[idx]
    neighbors_label = labels[idx]

    neighbors =  {'data': neighbors_data[:K], 'labels': neighbors_label[:K]}
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    #Assign the votes for every class
    for i in range(len(neighbors)):
        response = neighbors[i]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i] == predictions[i]:
            correct += 1
    return (float(correct)/float(len(testSet))) * 100.0

def predictKNN(dataframe, colname, predictor_names, K=10, train_size = 0.7): #optional training size
    training_testing_data = dataframe[dataframe[colname].notnull()]
    train, test = train_test_split(training_testing_data, train_size = train_size)
    train_X = np.array(train[predictor_names])
    train_Y = np.array(train[colname])
    test_X = np.array(test[predictor_names])
    test_Y = np.array(test[colname])

    predict_data = dataframe[dataframe[colname].isnull()]
    predict_X = np.array(predict_data[predictor_names])

    predictions = []
    for i in range(len(test_Y)):
        neighbors = getNeighbors(train_X, train_Y, test_X[i,:], K)
        result = getResponse(neighbors['labels'])
        predictions.append(result)
    accuracy = getAccuracy(test_Y, predictions)
    print ('Accuracy: ', accuracy, '%')

    predict_Y = []
    for i in range(len(predict_X)):
        neighbors = getNeighbors(train_X, train_Y, predict_X[i,:], K)
        result = getResponse(neighbors['labels'])
        predict_Y.append(result)

    predict_data[colname] = predict_Y
    dataframe = training_testing_data.append(predict_data)
    return dataframe[colname]

def replace_null_with_knn(dataframe, colname, predictor_names):
    column_new = pd.Series(predictKNN(dataframe, colname, predictor_names))
    column_new.name = colname + '_knn'
    return column_new