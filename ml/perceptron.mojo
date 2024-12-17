import numojo as nm
from numojo.prelude import *

struct Perceptron[dtype: DType = DType.float64]():
    var lr: Scalar[dtype]
    var iterations: Int 
    var weights: NDArray[dtype] 
    var bias: Scalar[dtype]

    fn __init__(inout self, lr: Scalar[dtype] = 1e-3, iterations: Int = 1000) raises:
        self.lr = lr
        self.iterations = iterations
        self.bias = 0.0
        self.weights = nm.zeros[dtype](Shape(1))

    fn fit(inout self, X: NDArray[dtype], y: NDArray[dtype]) raises:
        self.weights = nm.zeros[dtype](Shape(X.shape[1], 1))

        for _ in range(self.iterations):
            for i in range(X.shape[0]):
                var dw = self.lr * (y.get(i) - self.predict(X[i]).get(0))
                var temp = X[i]
                temp.reshape(X.shape[1], 1)
                self.weights += dw * temp
                self.bias += dw

    fn predict(self, X: NDArray[dtype], test: Bool = False) raises -> NDArray[dtype]:
        var linear = nm.matmul_parallelized[dtype](X, self.weights)
        linear += self.bias
        if test:
            print("X: ", X)
            print("Weights: ", self.weights)
            print("linear: ", linear)
        var mask = linear > 1.0
        var mask2 = linear < 0.0
        nm.where[dtype](linear, 1.0, mask) # modify numojo where function
        nm.where[dtype](linear, 0.0, mask2)
        return linear