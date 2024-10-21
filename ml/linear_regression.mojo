import numojo as nm
from numojo import *
from math import sqrt

from utility import r2_score, corrcoef

struct LinearRegression[dtype: DType = DType.float32]: 
    var lr: Scalar[dtype] 
    var n_iters: Int
    var weights: NDArray[dtype]
    var bias: Scalar[dtype]

    fn __init__(inout self, learning_rate: Scalar[dtype] = 0.001, n_iters: Int = 1000, shape: Int = 10) raises:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = nm.zeros[dtype](shape)  # Initialize weights with zeros
        self.bias = 0.0

    fn fit(inout self, inout X: NDArray[dtype], y: NDArray[dtype]) raises:
        var n_samples = X.ndshape[0]    
        var y_predicted: NDArray[dtype] 
        var dw: Scalar[dtype] = 0.0
        var db: Scalar[dtype] = 0.0

        for _ in range(self.n_iters):
            y_predicted = X * self.weights
            y_predicted += self.bias

            dw = nm.cumsum((y_predicted - y) * X) / n_samples
            db = nm.cummean(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias = self.bias - self.lr * db

    fn predict(self, X: NDArray[dtype]) raises -> NDArray[dtype]:
        var result = nm.dot(X, self.weights) 
        result += self.bias
        return result

