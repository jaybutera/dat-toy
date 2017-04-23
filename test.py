import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.layers import LSTM, Dense, Conv1D, Dropout
from keras.models import Model, Sequential
from common import *



winsize = 30
test_ratio = .9

X_train, Y_train, X_test, Y_test, fuels = load_data(winsize, test_ratio)

model = Sequential()
#model.add( Conv1D(64,4, input_shape=(None, 3)))#, batch_input_shape=(32,winsize,3)) )
#model.add( Conv1D(32,4) )
#model.add( Dropout(24) )
model.add( Dense(64, input_shape=(None, 3)) )#, batch_input_shape=(32,winsize,1)) )
#model.add( LSTM(32, input_shape=(None,3)) )
#model.add( LSTM(32, stateful=False) )
model.add( Dense(32, activation='relu') )
model.add( Dense(64, activation='relu') )
model.add( Dense(winsize, activation='relu') )

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['acc'])
print(model.summary())

model.fit(X_train, Y_train, epochs=2, shuffle=True)

#save_model(model)
model.save('model.hdf5')

preds = model.predict(X_test)

plot_res(preds, fuels, winsize)
