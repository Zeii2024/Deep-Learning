'''
using keras build simple GRU model

author @Terry
Feb 8 2020 
'''

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorboard

from keras.models import Sequential, Model
# from keras.utils import to_categorical
from keras.layers import Dense,Input, Dropout, Embedding, GRU, Flatten
from keras.callbacks import EarlyStopping

def GRUnet():

    model = Sequential()
    model.add(Embedding(3800, 32, input_length=380))
    model.add(Dropout(0.2))
    model.add(GRU(32))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

model = GRUnet()
# output parameters
model.summary()

# compile
model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

# train data
x_train, y_train = [], []
x_test, y_test = [], []
es = EarlyStopping(monitor='val_acc', patience=5)

# fit
batch_size = 32
epochs = 20
model.fit(x_train, y_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=epochs,
            callback=[es],
            shuffle=True)

# evaluate
scores = model.evaluate(x_test, y_test)
print("loss = {a}, acc = {b}".format(scores[0], scores[1]))
