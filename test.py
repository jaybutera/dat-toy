import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.layers import LSTM, Dense, Conv1D, Dropout, Conv2D
from keras.models import Model, Sequential
from common import *

winsize = 30
test_ratio = .9


model = Sequential()
#model.add( Conv1D(64,4, input_shape=(None, 7)))#, batch_input_shape=(32,winsize,3)) )
#model.add( Conv1D(32,4) )
#model.add( Dropout(24) )
model.add( Dense(64, input_shape=(None, 7)) )#, batch_input_shape=(32,1,7)) )
#model.add( LSTM(32, input_shape=(None,3)) )
#model.add( LSTM(32, stateful=False) )
model.add( Dense(24, activation='relu') )
model.add( Dense(64, activation='relu') )
model.add( Dense(48, activation='relu') )
model.add( Dense(winsize, activation='sigmoid') )

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['acc'])
print(model.summary())

files = ['20170303_sec.csv', '20170307_sec_trip2.csv',
        '20170308_sec_trip3.csv']

#X_tr = np.array([], dtype=np.float32)
#Y_tr = np.array([])
#X_te = np.array([])
#Y_te = np.array([])
X_tr, Y_tr, X_test, Y_test, fuels = load_data(winsize, test_ratio, files[0])

for f in files:
    if f != files[0]:
        X_train, Y_train, X_test, Y_test, fuels = load_data(winsize, test_ratio, f)

        #X_tr += X_train
        #Y_tr += Y_train
        np.concatenate((X_tr, X_train), axis=0)
        np.concatenate((Y_tr, Y_train), axis=0)
    #np.concatenate(X_te, X_test)
    #np.concatenate(Y_te, Y_test)

print(X_tr.shape)
model.fit(X_tr, Y_tr, epochs=10, shuffle=True)

#save_model(model)
model.save('model.hdf5')

#preds = model.predict(X_test)

#plot_res(preds, fuels, winsize)
