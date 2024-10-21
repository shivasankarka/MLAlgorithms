from math import sqrt, exp
from numojo import *
import numojo as nm

fn accuracy_score[dtype: DType](y_test: NDArray[dtype], y_pred: NDArray[dtype]) raises -> Scalar[f64]:
    var correct_count: Scalar[f64] = 0.0
    for i in range(y_test.ndshape.ndsize):
        if y_test.data[i] == y_pred.data[i]:
            correct_count += 1.0
    return correct_count / y_test.ndshape.ndsize

fn sigmoid[dtype: DType](x: NDArray[dtype]) raises -> NDArray[dtype]:
    var temp: NDArray[dtype] = zeros[dtype](x.ndshape)
    for i in range(x.ndshape.ndsize):
        temp.store[width=1](i, val= 1 / (1 + exp(-x.load[width=1](i))))
    return temp^

fn mse[dtype: DType](y_true: NDArray[dtype], y_pred: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    var n = y_true.ndshape[0]
    var diff = y_true - y_pred
    var squared_diff = diff * diff
    var sum_squared_diff = squared_diff.cumsum()
    return sum_squared_diff / Scalar[dtype](n)

fn corrcoef[dtype: DType](x: NDArray[dtype], y: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    var n = x.ndshape[0]
    var mean_x = x.cumsum() / n
    var mean_y = y.cumsum() / n

    var numerator: Scalar[dtype] = Scalar[dtype](0)
    var denominator_x: Scalar[dtype] = Scalar[dtype](0)
    var denominator_y: Scalar[dtype] = Scalar[dtype](0)

    for i in range(n):
        var diff_x = x.get(i) - mean_x
        var diff_y = y.get(i) - mean_y
        numerator += diff_x * diff_y
        denominator_x += diff_x * diff_x
        denominator_y += diff_y * diff_y

    return numerator / (sqrt(denominator_x) * sqrt(denominator_y))

fn r2_score[dtype: DType](y_true: NDArray[dtype], y_pred: NDArray[dtype]) raises -> SIMD[dtype, 1]:
    var corr = corrcoef(y_true, y_pred)
    return corr * corr

fn transpose[dtype: DType](X: NDArray[dtype]) raises -> NDArray[dtype]:
    var X_T: NDArray[dtype] = zeros[dtype](shape(X.ndshape[1], X.ndshape[0]))
    for i in range(X.ndshape[0]):
        for j in range(X.ndshape[1]):
            X_T[idx(j, i)] = X[idx(i, j)]
    return X_T^
