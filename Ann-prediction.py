# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#part 1 - data preprocessiong 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


from keras.models import Sequential 
from keras.layers import Dense  
from keras.layers import Dropout 
# Initializing the ANN 
classifier = Sequential()
# Adding the i/p layer and the first hidden layer 
classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu', 
                     input_dim = 18))
# Adding the 2nd Hidden layer 
classifier.add(Dropout(0.2))
classifier.add(Dense(output_dim = 512, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))


classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu'))


classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 64, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
#  Adding the output layer 
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# compiling the ANN 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])
# Fitting the ANN in to the training set 2
classifier.fit(x_train, y_train, batch_size = 512, nb_epoch = 100)

# making the predictions and evaluating the model 
# fitting the classifier to the training set 
# create your classifier here 
# predicting the test set result 
y_pred = classifier.predict(x_test1) 
np.max(y_pred)
y_pred = (y_pred > 0.08)
# making the confusion matrix

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)

# calculating the total accuracy (85.9%)
(1516+202)/2000