from numojo import *
from numojo.core.array_manipulation_routines import where

from .utility import sigmoid, transpose

struct LogisticRegression[dtype: DType = DType.float64]:
    var lr: Scalar[dtype]
    var n_iters: Int
    var weights: NDArray[dtype]
    var bias: Scalar[dtype]

    fn __init__(inout self, n_features: Int, learning_rate: Scalar[dtype] = 0.001, n_iters: Int = 1000) raises:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.bias = 0.0
        self.weights = zeros[dtype](shape(n_features))

    fn fit(inout self, inout X: NDArray[dtype], y: NDArray[dtype]) raises:
        var n_samples: Scalar[dtype] = Scalar[dtype](X.ndshape[0])

        for i in range(self.n_iters):
            var temp: NDArray[dtype] = matmul_naive[dtype](X, self.weights)
            temp += self.bias
            var y_predicted: NDArray[dtype] = sigmoid(temp)
            var transposed_X = transpose[dtype](X)
            var dw = matmul_naive[dtype](transposed_X, y_predicted-y) / n_samples
            var db = cummean(y_predicted - y) / n_samples
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    fn predict(self, X: NDArray[dtype]) raises -> NDArray[i16]:
        var linear_model = matmul_naive[dtype](X, self.weights)
        linear_model += self.bias
        var classes: NDArray[i16] = self.predict_class(linear_model, gt=True)
        return classes^

    fn predict_class(self, X: NDArray[dtype], gt: Bool) raises -> NDArray[i16]:
        var classes: NDArray[i16] = zeros[i16](X.ndshape)
        for i in range(X.ndshape.ndsize):
            if X.get(i) > 0.5:
                classes.set(i, 1)
            else:
                classes.set(i, 0)
        return classes^
