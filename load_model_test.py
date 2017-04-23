from keras.models import Model
import common
from keras.models import load_model

model = load_model('model.hdf5')

winsize = 30
test_ratio = .9
X_train, Y_train, X_test, Y_test, fuels = common.load_data(winsize, test_ratio)

preds = model.predict(X_test)

common.plot_res(preds, fuels, winsize)
