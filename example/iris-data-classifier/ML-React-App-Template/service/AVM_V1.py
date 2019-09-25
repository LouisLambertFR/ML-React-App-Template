import os
import sys
import time as t
import numpy as np
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

#Pipeline Step 1: Data Preparation

def preProcessDataset():

    # Imports our data into a dataframe
    f = open('Clean_data_set_2018.csv')
    dataframe = pd.read_csv(f)

    # Prepare the dataset for training/testing
    prepareDataset(dataframe)

def prepareDataset(dataframe):

    # Drop specified columns
    dataframe.drop(columns=['Date mutation','No voie','Type de voie','Code voie','Voie','Code postal','Code postal','Commune','Code departement','Code commune','Section','No plan','Type local'], inplace=True)

    # Create X/Y datasets
    x = dataframe.iloc[:, 1:3]
    y = dataframe.iloc[:, 0]

    # Split train and test sets
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=1)

    # Start training
    startTraining(xtrain, xtest, ytrain, ytest)

def startTraining(xtrain, xtest, ytrain, ytest):

    # Train the model
    regressor = SVR(kernel='linear', degree=1)
    regressor.fit(xtrain, ytrain)
    y_pred = regressor.predict(xtest)

    # Start evaluating model performance
    calculatePerformance(ytest,y_pred)

def calculatePerformance(y_test, y_pred):

    # mean_absolute_percentage_error (MPE)
    y_true, y_pred = np.array(y_test), np.array(y_pred)
    medianpercentageerror = np.median(np.abs((y_true - y_pred) / y_true)) * 100
    meanpercentageerror = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print('Median Percentage Error ~ ' + str(round(medianpercentageerror)) + ' %')
    print('Mean Percentage Error ~ ' + str(round(meanpercentageerror)) + ' %')

if __name__ == '__main__':
    start = t.process_time()
    relay = preProcessDataset()
    print('Program executed in ~ ' + str(round(t.process_time() - start)) + ' seconds')