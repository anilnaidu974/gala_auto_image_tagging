import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# features path
features_path = './features/features.pickle'
labels_path = './features/labels.pickle'

def train_model(trainX, trainY):
    # train the on different parameters and returning the best one
    params = {"C":[0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]}
    model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1, verbose=1)
    model.fit(trainX,trainY)
    return model

def evaluate_model(model, testX, testY):
    y_pred=model.predict(testX)
    print(classification_report(testY,y_pred, target_names = ['Food','Attire','Decoration','misc']))


def main():
    # reading pickle files
    features = pd.read_pickle(features_path)
    labels = pd.read_pickle(labels_path)

    #split the data into test and train
    (trainX, testX, trainY, testY) = train_test_split(features,
	    labels, test_size=0.2, random_state=42)
    
    # train the model on the features
    model = train_model(trainX, trainY)

    # evaluating the model performance
    evaluate_model(model, testX, testY)

    # storing the model
    pickle.dump(model, open('./model/regression.sav', 'wb'))
 

if __name__ == "__main__":
    main()
