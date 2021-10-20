'''
Data set introduction
The data consists of 48x48 pixel grayscale images of faces
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The faces have been automatically registered so that the face is more or less centered
and occupies about the same amount of space in each image
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def main():
    ''' ### Read csv data '''
    df = pd.read_csv('fer2013\\train.csv')
    print("There are total ", len(df), " sample in the loaded dataset.")
    print("The size of the dataset is: ", df.shape)
    # get a subset of the whole data for now
    df = df.sample(frac=0.1, random_state=46)
    print("The size of the dataset is: ", df.shape)



    ''' Extract images and label from the dataframe df '''
    width, height = 48, 48
    images = df['pixels'].tolist()
    faces = []
    for sample in images:
        face = [int(pixel) for pixel in sample.split(' ')]  # Splitting the string by space character as a list
        face = np.asarray(face).reshape(width*height)       # convert pixels to images and # Resizing the image
        faces.append(face.astype('float32') / 255.0)       # Normalization
    faces = np.asarray(faces)

    # Get labels
    y = df['emotion'].values


    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Visualization a few sample images
    plt.figure(figsize=(5, 5))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(faces[i].reshape(width, height)), cmap='gray')
        plt.xlabel(class_names[y[i]])
    plt.show()

    ## Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(faces, y, test_size=0.1, random_state=46)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    allModels = []
    # allModels.append(makeSVCClassifier(X_train, y_train))
    # allModels.append(makeRandomForest(X_train, y_train))
    allModels.append(makeDecisionTreeClassifer(X_train, y_train))

    for model in allModels:
        # Now that our classifier has been trained, let's make predictions on the test data. To make predictions, the predict method of the DecisionTreeClassifier class is used.
        y_pred = model.predict(X_test)

        # For classification tasks some commonly used metrics are confusion matrix, precision, recall, and F1 score.
        # These are calculated by using sklearn's metrics library contains the classification_report and confusion_matrix methods
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

def makeSVCClassifier(X_train, y_train):
    svclassifier = SVC(kernel='linear')
    
    svclassifier.fit(X_train, y_train)

    return svclassifier

def makeRandomForest(X_train, y_train):
    rfc = RandomForestClassifier(n_estimators=100)

    rfc.fit(X_train,y_train)

    return rfc

def makeDecisionTreeClassifer(X_train, y_train):
    dtc = DecisionTreeClassifier()

    dtc.fit(X_train, y_train)

    return dtc

main()