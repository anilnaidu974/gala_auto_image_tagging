# python features_extraction.py --images './dataset/Train Images/' --csv './dataset/train.csv'

import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import pickle
import argparse

from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder

# image dimensions
WIDTH = 150
HEIGHT = 150
DEPTH = 3

# dataset paths to read images
class_label={'Food':0,'Attire':1,'Decorationandsignage':2,'misc':3}

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="folder path to train images")

ap.add_argument("-c", "--csv", required=True,
	help="path to csv files")
args = vars(ap.parse_args())

# deep learning model to extract the features fro  image
resNet = ResNet50(include_top=False, weights='imagenet',input_shape=(WIDTH,HEIGHT,DEPTH),pooling='avg')

def preprocess(image_path):
    # read the image and return preprocessed image
    image = load_img(image_path, target_size=(WIDTH,HEIGHT))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image

def extract_features(train_img):
    print("--------features extraction---------")
    features = resNet.predict(train_img, verbose=1)
    return features

def store_features(features,labels):
    # storing extracted features
    print("--------features storing---------")
    with open('./features/features.pickle', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./features/labels.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main(train_images_path, train_csv_path):
    # reading training csv file
    train=pd.read_csv(train_csv_path)
    train['Class']=train['Class'].map(class_label)
    train_img=[]
    train_label=[]
    j=0
    # reading all images into array
    for i in tqdm(train['Image']):
        image_path=os.path.join(train_images_path,i)
        image =  preprocess(image_path)
        train_img.append(image)
        train_label.append(train['Class'][j])
        j=j+1
    
    train_img = np.vstack(train_img)
    train_label=np.array(train_label)
    # features Extraction
    features = extract_features(train_img)
    # storing
    store_features(features,train_label)

    print("--------completed---------")

if __name__ == "__main__":
    train_images_path = args["images"]
    train_csv_path = args["csv"]
    main(train_images_path, train_csv_path)
