#import python classes and packages 
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential, load_model 
#loading CNN3D classes 
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, InputLayer, 
BatchNormalization, Dropout, GlobalAveragePooling3D, MaxPooling2D 

from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from keras.callbacks import ModelCheckpoint 
import pandas as pd 
from keras import layers 
import numpy as np 
import keras 
import os 
from keras.layers import Convolution2D 
import pickle 
from keras.layers import Bidirectional, GRU, Conv1D, MaxPooling1D, RepeatVector#loading 
GRU, bidriectional, and CNN 
import seaborn as sns 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt 
#defining class labels 
labels = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Laying'] 
#loading UCI Har dataset captured activities from smart phones 
X = pd.read_csv("Dataset/X_train.txt", header=None, delim_whitespace=True) 
 
Y = pd.read_csv("Dataset/y_train.txt", header=None, delim_whitespace=True) 
X#visualizing class labels count found in dataset 
names, count = np.unique(Y, return_counts = True) 
height = count 
bars = labels 
y_pos = np.arange(len(bars)) 
#train existing CNN algorithm which will use many parameters for training and can increase 
computation complexity 
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 
(X_train.shape[3] * X_train.shape[4]))) 
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 
(X_test.shape[3] * X_test.shape[4])))  
cnn_model = Sequential() 
#define cnn2d layer with 3 number of inout neurons and to filter dataset features 
cnn_model.add(Convolution2D(3, 
(1 
, 
1), 
input_shape 
X_train1.shape[2], X_train1.shape[3]), activation = 'relu')) 
#collect filtered features from CNN2D layer 
cnn_model.add(MaxPooling2D(pool_size = (1, 1))) 
#defining another layer t further optimize features 
cnn_model.add(Convolution2D(3, (1, 1), activation = 'relu')) 
cnn_model.add(MaxPooling2D(pool_size = (1, 1))) 
cnn_model.add(Flatten()) 
#define output layer 
= (X_train1.shape[1], 
cnn_model.add(Dense(units = 16, activation = 'relu')) 
cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax')) 
#compile and train the model 
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 
['accuracy']) 
x = layers.MaxPool3D(pool_size=1)(x) 
x = layers.BatchNormalization()(x) 
x = layers.Conv3D(filters=7, kernel_size=1, activation="relu")(x) 
x = layers.MaxPool3D(pool_size=1)(x) 
x = layers.BatchNormalization()(x) 
x = layers.Conv3D(filters=32, kernel_size=1, activation="relu")(x)#cnn layer for separable 
convolution module 
x = layers.MaxPool3D(pool_size=1)(x) 
x = layers.BatchNormalization()(x) 
x = layers.GlobalAveragePooling3D()(x)#defining global average pooling 
x = layers.Dense(units=64, activation="relu")(x) 
x = layers.Dropout(0.3)(x) 
outputs = layers.Dense(units=y_train.shape[1], activation="softmax")(x) 
mdn_model = keras.Model(inputs, outputs, name="3dcnn") #create model 
mdn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 
['accuracy']) 
#displaying propose model complexity 
print(mdn_model.summary())extension_model.add(Dense(units = 1, activation = 'relu')) 
extension_model.add(Dense(units = y_train.shape[1], activation = 'softmax')) 
#compile and train the model 
extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 
['accuracy']) 
print(extension_model.summary())  
if os.path.exists("model/extension_weights.hdf5") == False: 
    model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose 
= 1, save_best_only = True) 
    hist = extension_model.fit(X_train, y_train, batch_size = 32, epochs = 20, 
validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1) 
    f = open('model/extension_history.pckl', 'wb') 
    pickle.dump(hist.history, f) 
    f.close()     
else: 
    extension_model = load_model("model/extension_weights.hdf5") 
#perform prediction on test data using extension model 
predict = extension_model.predict(X_test1) 
predict = np.argmax(predict, axis=1) 
y_test1 = np.argmax(y_test, axis=1)
