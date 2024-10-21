from ml.linear_regression import LinearRegression
from ml.utility import r2_score, corrcoef, mse

from python import Python
from numojo import *

def main():
    Python.add_to_path(".")
    lr_test = Python.import_module("test_linearReg")
    var np = Python.import_module("numpy")
    data = lr_test.get_data() # X, X_train, X_test, y_train, y_test

    var X_train = array(data = data[1])
    var X_test =  array(data = data[2])
    var y_train =  array(data = data[3])
    var y_test =  array(data = data[4])

    lr = LinearRegression(0.001, 3000, shape=data[1].shape[0])
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("MSE:", mse(y_test, y_pred))
    print("Accuracy:", r2_score(y_test, y_pred))

    var test_X = np.linspace(-2, 2, 20)
    y_pred_line = lr.predict(test_X)
    lr_test.test(test_X, data[1], data[2], data[3], data[4], y_pred_line.to_numpy())
