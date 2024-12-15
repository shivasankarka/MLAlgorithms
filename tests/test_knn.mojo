from python import Python, PythonObject
import time

import numojo as nm
from ml.knn import KNN

def main():
    knn_test = Python.import_module("test_knn")
    data = knn_test.get_data()
    X_train, X_test, y_ = train_test_split(nm.array(data=data[0]), data[1], test_size=0.2, random_state=1234)
    knn = KNN(k = 3)
    knn.fit(X_train, y_.train)
    y_pred = knn.predict(X_test)
    print("KNN classification accuracy:", accuracy_score(y_.test, y_pred))

fn train_test_split[dtype: DType](X: NDArray[dtype], y: PythonObject, test_size: Float16 = 0.5, random_state: Int = time.perf_counter_ns()) raises -> Tuple[, Matrix, SplittedPO]:
    var np = Python.import_module("numpy")
    var ids = Matrix.rand_choice(X.height, X.height, False, random_state)
    var split_i = int(X.height - (test_size * X.height))
    var y_train = np.empty(split_i, dtype='object')
    var y_test = np.empty(X.height - split_i, dtype='object')
    for i in range(split_i):
        y_train[i] = y[ids[i]]
    for i in range(split_i, X.height):
        y_test[i - split_i] = y[ids[i]]
    return X[ids[:split_i]], X[ids[split_i:]], SplittedPO(y_train, y_test)
