'''
using keras build simple RNN model

author @Terry
Jan 22 2020 
'''

import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense,Input, Dropout, Embedding, LSTM, Bidirectional, Flatten
from keras.layers.recurrent import SimpleRNN
from keras.callbacks import EarlyStopping

def RNN(maxlen=380, max_features=3800, embed_size=32):
    '''
    Single layer RNN (one SimpleRNN)
    # Embedding(input_dim, output_dim, input_length)
    # Embedding层放在第一层，将输入数据的稀疏矩阵映射到密集矩阵，比如
    # 输入信息为语句或单词，将其变为向量
    # 有三个参数，input_dim，输入的数据数量
    # output_dim，输出数据的维度；input_length，每条数据的长度，后面接Flatten层时
    # 需要指定该参数
    # Dropout是神经元失活比例，0.5表示每次随机50%神经元不训练，防止过拟合
    '''
    model = Sequential()

    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(SimpleRNN(16))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def BRNN(maxlen=380, max_features=3800, embed_size=32):
    '''
    Bidirectory RNN
    双向RNN，由正向和反向两个RNN组成，分别处理正向和反向的信息
    每个单元需要将正反两向的重要信息结合起来，从而达到结合上下方预测的目的
    于是SimpleRNN中return_sequences要设置为True，正反向RNN将输出结果返回整个
    序列，而后合并concat(/sum/mul/aver/None)
    '''
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen))
    model.add(Dropout(0.5))
    model.add(Bidirectional(SimpleRNN(16, return_sequences=True),merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Flatten()) # 全连接层
    model.add(Dense(1, activation='sigmoid'))

    return model

model = RNN()
# model = BRNN()
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
verbose = 0: 信息显示的模式
mode：有'min', 'max', 'auto',指当监测指标不再减小或增大时，停止训练
'''
# need train and test data
x_test, y_test, x_train, y_train = 0, 0, 0, 0
batch_size = 128
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