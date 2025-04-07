#-------------------------------------------------------------------------
# AUTHOR: Anh Tu Nguyen
# FILENAME: perceptron.py
# SPECIFICATION: This Python script implements a comparative study between a single-layer Perceptron and a Multi-Layer Perceptron (MLP) 
# for classifying optically recognized handwritten digits. 
# The code uses the Python libraries NumPy and pandas to handle data and scikit-learn for building and evaluating the models.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# Read the training data from optdigits.tra
df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

#Initialize best accuracy trackers forr both algorithnms
best_perc_acc = 0
best_mlp_acc = 0

#Iterate over learning rates
for lr in n: 
    #Iterate over shuffle options:
    for sh in r: #iterates over r
        #iterates over both algorithms
        for algo in ["Perceptron", "MLP"]:
            
            #Create the classifier based on the algorithm name
            if algo == "Perceptron":
                clf = Perceptron(eta0=lr, shuffle=sh, max_iter=1000, random_state=42)
            else: #MLP classifier
                clf = MLPClassifier(activation='logistic', learning_rate_init=lr, hidden_layer_sizes=(25,), 
                                    shuffle=sh, max_iter= 1000, random_state=42)
            #Fit the classifier to the training data
            clf.fit(X_training, y_training)

            # Make predictions for each test sample individually using zip()
            correct_count = 0
            for x_testSample, y_testSample in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])[0]
                if prediction == y_testSample:
                    correct_count += 1
            acc = correct_count / len(y_test)
            
            #Check if the accuracy is higher than previously recorded; update and print message if so
            if algo == "Perceptron":
                if acc > best_perc_acc:
                    best_perc_acc = acc
                    print("\n")
                    print(f"Highest Perceptron accuracy so far: {acc:.4f}, Parameters: learning rate={lr}, shuffle={sh}")
                    print("\n")
            else: #MLP
                if acc > best_mlp_acc:
                    best_mlp_acc = acc
                    print(f"Highest MLP accuracy so far: {acc:.4f}, Parameters: learning rate={lr}, shuffle={sh}")
                    print("\n")
            











