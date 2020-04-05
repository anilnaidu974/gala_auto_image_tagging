# python evaluate.py --images './dataset/Test Images/' --csv './dataset/test.csv' --output './results/data.csv'

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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="folder path to test images")
ap.add_argument("-c", "--csv", required=True,
	help="path to csv files")
ap.add_argument("-o", "--output", default='./results/data.csv',
	help="path to csv files")
args = vars(ap.parse_args())


# image dimensions
WIDTH = 150
HEIGHT = 150
DEPTH = 3

# model path
model_path = './model/regression.sav'
regression_model = pickle.load(open(model_path, 'rb'))

class_to_label={0:'Food',1:'Attire',2:'Decorationandsignage',3:'misc'}

# features path
features_path = './features/features.pickle'
labels_path = './features/labels.pickle'

# deep learning model
resNet = ResNet50(include_top=False, weights='imagenet',input_shape=(150,150,3),pooling='avg')

def preprocess(image_path):
    # read the image and return preprocessed image
    image = load_img(image_path, target_size=(WIDTH,HEIGHT))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image

def save_test_results(image_name,class_name, output):
    df = pd.DataFrame(data={"Image": image_name, "Class": class_name})
    df.to_csv(output, sep=',',index=False)


def main(images_path, csv_path, output):

    image_name = []
    class_name = []
    test = pd.read_csv(csv_path)
    print('-------------- Processing---------------')
    for i, row in test.iterrows():            
        # Images
        image_path=os.path.join(images_path,row['Image'])
        image = preprocess(image_path)
        feature = resNet.predict(image)
        image_name.append(row['Image'])
        pred = regression_model.predict(feature)
        class_name.append(class_to_label[pred[0]])
    
    save_test_results(image_name,class_name, output)
    print("---------- Evaluation on test data completed-------------")
 

if __name__ == "__main__":
    images_path = args["images"]
    csv_path = args["csv"]
    output = args["output"]
    main(images_path,csv_path,output)
