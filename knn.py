import numpy as np
import scipy as sp
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt


# load dataset
dataset = scio.loadmat('dataset.mat')

# get training and testing sets
x_train = dataset['train_image']
x_test = dataset['test_image']
y_train = dataset['train_label']
y_test = dataset['test_label']

accuracy = 0

def dist(xTest, yTest, K):
    distSum = 0
    score = [[-1 for x in range(2)] for y in range(len(x_train))]

    # calculate distance
    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            for k in range(len(x_train[i][j])):
                distSum += abs(xTest[j][k] - x_train[i][j][k])
        score[i][0] = distSum
        score[i][1] = y_train[i][0]
        distSum = 0

    # sort list by distance value
    sortedScore = sorted(score,key=lambda x: (x[0],x[1]))
    
    # return top K results
    zeros, ones = 0, 0
    for i in range(0, K):
        ones += sortedScore[i][1]
    zeros = K-ones
    
    # give prediction
    prediction = None

    if zeros == ones:
        # calculate total distance for 0 and 1
        sumZeros, sumOnes = 0,0
        for i in range(0, K):
            if sortedScore[i][1] == 0:
                sumZeros += sortedScore[i][0]
            else:
                sumOnes += sortedScore[i][0]
        
        # give prediction based on smallest total distance
        if sumZeros < sumOnes:
            prediction = 0
        elif sumZeros > sumOnes:
            prediction = 1
        else:
            prediction = sortedScore[0][1]

    elif zeros > ones:
        prediction = 0
    elif zeros < ones:
        prediction = 1

    # compare with y_test
    if prediction == yTest:
        global accuracy 
        accuracy += 1

array = [None] * 11
for i in range(1,11):
    for j in range(0, len(x_test)):
        dist(x_test[j], y_test[j], i)
    print ("Accuracy for K = ", i, ":", accuracy/len(y_test))
    array[i] = accuracy/len(y_test)
    accuracy = 0
print (array)
plt.plot(array, 'ro')
plt.xlabel('Number of nearest neighbour')
plt.ylabel('Accuracy')
plt.show()