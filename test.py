import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing as pp

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.layers import LSTM, Dense, Conv1D, Dropout
from keras.models import Model, Sequential

import csv

def load_file(filename):
    with open(filename) as f:
        r = csv.DictReader(f)
        data = []
        for row in r:
            data.append(row)

    return data
    #reader = list( csv.reader( open(filename, 'r'), delimiter=',') )

    return np.array(reader)#.astype('float')

def windowfy (l, winsize):
    return np.array([l[i:i+winsize] for i in range(len(l) - winsize)])

def floatify (l):
    d = []
    for x in l:
        try:
            d.append( float(x) )
        except ValueError:
            print('invalid string to float: ', x)
            d.append(0.)

    return d

def load_data (winsize, test_ratio):
    data = load_file('20170303_sec.csv')

    mms = pp.MinMaxScaler()

    cut = int(len(data) * test_ratio)

    speeds = mms.fit_transform( [float(x['Vehicle_Speed']) for x in data[:-2*winsize]] )
    brakes = mms.fit_transform( floatify([x['Brake_Control_Volume'] for x in data[:-2*winsize]]) )
    engs   = mms.fit_transform( floatify([x['Engine_Speed'] for x in data[:-2*winsize]]) )

    fuels  = mms.fit_transform( floatify([x['Fuel_Consum'] for x in
        data[winsize:-winsize]]) )

    #speeds_w = windowfy(speeds, winsize)
    #speeds_w = speeds_w.reshape( len(speeds_w), winsize, 1)
    #brakes_w = windowfy(brakes, winsize)
    #engs_w   = windowfy(engs, winsize)
    #engs_w = engs_w.reshape( len(engs_w), winsize, 1)

    fuels_w  = windowfy(fuels, winsize)
    fuels_w = fuels_w.reshape( len(fuels_w),1, winsize)

    #X_train = np.array([speeds_w, brakes_w, engs_w, fuels_w])
    #X = np.array([speeds_w, engs_w, brakes_w])
    #X = X.reshape(speeds_w.shape[0], winsize, 3)
    #X = np.transpose(X, axes=(1,2,0))
    X = np.transpose( np.array([speeds, engs, brakes]) )
    X = X.reshape(X.shape[0], 1, X.shape[1])

    X_train = X[:cut]
    #X_train = np.array(engs_w[:cut])
    Y_train = np.array(fuels_w[:cut])
    X_test  = X[-cut:]
    #X_test  = np.array(engs_w[-cut:])
    Y_test  = np.array(fuels_w[-cut:])

    return X_train, Y_train, X_test, Y_test, fuels[-cut:]

def plot_res (preds, Y, winsize):
    t = np.linspace(0, len(preds)-winsize, int(len(Y) ), dtype=int)
    num = int(len(preds) / winsize)

    #plt.plot([preds[i] for i in t], t, 'r')
    tmp = []
    for i in range(num):
        tmp += preds[i*winsize][0].tolist()
    print('tmp len: ', len(tmp))

    for i in range(len(tmp)):
        if tmp[i] == 0. and i > 0:
            tmp[i] = tmp[i-1]

    plt.plot(tmp, 'r', marker='o')
    plt.plot(Y, 'b')
    plt.show()

def ma (l):
    return np.concatenate(np.array([ (l[i]+l[i+1])/2 for i in range(len(l-1))]), l[-1])


winsize = 30
test_ratio = .9

X_train, Y_train, X_test, Y_test, fuels = load_data(winsize, test_ratio)
print('X_test shape: ', X_test.shape)

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

model.fit(X_train, Y_train, epochs=400, shuffle=True)

preds = model.predict(X_test)

plot_res(preds, fuels, winsize)
