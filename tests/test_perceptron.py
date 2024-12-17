from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy

def generate_data():
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


    