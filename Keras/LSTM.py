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
from keras.callbacks import EarlyStopping
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

model = lstm()
# model = Bilstm()
# model.summary()

# Train
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc', patience=5, mode='auto')
'''
EarlyStopping 用于提前终止训练，是fit中callback的输入值之一；
monitor：需要监测的量, 有'val_loss', 'val_acc', 'acc', 'loss'
patience: 指可以容忍在多少个epoch内没有improvement
mode：有'min', 'max', 'auto',指当监测指标不再减小或增大时，停止训练
'''
# need train and test data
x_test, y_test, x_train, y_train = 0, 0, 0, 0
batch_size = 64
epochs = 20

# fit
model.fit(x_train, y_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[es],   # callbacks训练终止的条件，输入是list
            shuffle=True)


# evaluation
scores = model.evaluate(x_test, y_test)
print("loss: ", scores[0])
print("acc: ", scores[1])
