from python import Python
from numojo import *

from ml.logistic_regression import LogisticRegression
from ml.utility import accuracy_score

def main():
    Python.add_to_path(".")
    lr_test = Python.import_module("test_logisticReg")
    data = lr_test.get_data() # X_train, X_test, y_train, y_test

    var X_train = array(data=data[0])
    var y_train = array(data=data[2])
    var X_test = array(data=data[1])
    var y_test = array(data=data[3])

    lr = LogisticRegression[DType.float64](n_features=X_train.ndshape[1], learning_rate=0.0001, n_iters=1000)
    lr.fit(X_train, y_train)
    var y_pred: NDArray[i16] = lr.predict(X_test)
    var y_test_int: NDArray[i16] = y_test.astype[i16]()
    print("LogisticRegression classification accuracy:", accuracy_score[i16](y_test_int, y_pred))
