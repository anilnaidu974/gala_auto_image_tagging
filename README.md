# gala_auto_image_tagging

1. Feature extraction :
 
	run command - python features_extraction.py --images ‘./dataset/Train Images/’ --csv ‘./dataset/train.csv’

--images = path to the Train images
--csv = path to the train csv file

This file extract the features from train images and store pickle files of features and labels in ./features/ folder.

2. Train the model :

	run command - python train.py
It will take the features from the ./features/ folder and train on the features,
and it will save the model as ./model/regression.sav.


3. Evaluate the model :

	run command - python evaluate.py --images ‘./dataset/Test Images/’ --csv ‘./dataset/test.csv’

--images = path to the Train images.
--csv = path to the train csv file.

This python file extract the features on test images and predict the labels using the saved model.

Note : Model which i already trained using above approach is available in the ./model/ folder, you can directly jump to 3rd step (Evaluate the model) to test accuracy on Test Images.
