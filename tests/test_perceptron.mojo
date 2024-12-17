from ml.perceptron import Perceptron
from ml.utility import accuracy_score

import numojo as nm
from python import Python

fn test() raises:
    var data_sklearn = Python.import_module("test_perceptron")
    var data = data_sklearn.generate_data()
    X_train = nm.array[nm.f64](data[0])
    X_test = nm.array[nm.f64](data[1])
    y_train = nm.array[nm.f64](data[2])
    y_test = nm.array[nm.f64](data[3])

    print("X_train: ", X_train)
    print("y_train: ", y_train)

    p = Perceptron(lr=0.001, iterations=1000)
    p.fit(X_train, y_train)
    print("X_test: ", X_test)
    y_pred = p.predict(X_test, False)
    print("y_pred: ", y_pred)
    print("y_test: ", y_test)
    print("Perceptron classification accuracy:", accuracy_score[nm.f64](y_test, y_pred))
    # p_test.test(X_train.to_numpy(), y_train.to_numpy(), p.weights.to_numpy(), p.bias)

fn main() raises:
    test()