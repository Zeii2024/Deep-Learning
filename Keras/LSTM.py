'''
using keras build simple LSTM model

author @Terry
Jan 26 2020 
'''

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorboard

from keras.models import Sequential, Model
# from keras.utils import to_categorical
from keras.layers import Dense,Input, Dropout, Embedding, LSTM, Bidirectional, Flatten
# from keras.layers.recurrent import SimpleRNN

def lstm():
    '''
    简单的LSTM，类似于RNN
    '''
    model = Sequential()
    model.add(Embedding(3800, 32, input_length=380))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

def Bilstm():
    '''
    双向LSTM，类似于双向RNN
    '''
    model = Sequential()
    model.add(Embedding(3800, 32, input_length=380))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32, return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# model = lstm()
model = Bilstm()
model.summary()