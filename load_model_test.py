from keras.models import Model
import common
from keras.models import load_model
import numpy as np

model = load_model('model.hdf5')

def avg_cons (l):
    return np.average(l)

def eval (filenames):
    scores = []
    scores_pred = []
    for f in filenames:
        X_train, Y_train, X_test, Y_test, fuels = common.load_data(winsize, test_ratio, f)

        preds = model.predict(X_test)
        scores.append( np.average( np.array([np.average(x) for x in fuels]) ) )
        scores_pred.append( np.average( np.array([np.average(x) for x in preds]) ) )

    print('Target: ')
    print(scores)
    print('Predicted: ')
    print(scores_pred)

    return scores

winsize = 30
test_ratio = .9
filename = '20170303_sec.csv'
#X_train, Y_train, X_test, Y_test, fuels = common.load_data(winsize, test_ratio, filename)
#eval(['20170303_sec.csv', '20170307_sec_trip2.csv', '20170308_sec_trip3.csv'])


X_train, Y_train, X_test, Y_test, fuels = common.load_data(winsize, test_ratio, filename)
preds = model.predict(X_test)

common.plot_res(preds, fuels, winsize)
