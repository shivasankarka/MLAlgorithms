import numojo as nm
from numojo.prelude import *
from .utility import euclidean_distance

struct KNN[dtype: DType = f64]:
    var k: Int
    var train: NDArray[dtype]
    var test: NDArray[dtype]
    
    fn __init__(inout self, k: Int = 3) raises:
        self.k = k  
        self.train = NDArray[dtype](shape(1))
        self.test = NDArray[dtype](shape(1))
    
    fn fit(inout self, X_train: NDArray[dtype], y_train: NDArray[dtype]) raises:
        self.train = X_train
        self.test = y_train

    fn predict(self, X_test: NDArray[dtype]) raises -> NDArray[dtype]:
        var predictions: NDArray[dtype] = nm.zeros[dtype](shape(X_test.ndshape[0]))
        for i in range(X_test.ndshape[0]):
            var neighbors: List[Scalar[dtype]] = self.get_neighbors(X_test[i])
            var output_values: List[Scalar[dtype]] = [neighbors[j][1] for j in range(neighbors.len)]
            var prediction: Scalar[dtype] = self.get_response(output_values)
            predictions[i] = prediction
        return predictions^
