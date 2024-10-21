import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def get_data():
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    
    
    return [X, X_train, X_test, y_train, y_test]

def test(X, X_train, X_test, y_train, y_test, y_pred_line):
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    # plt.scatter(X, y_pred_line)
    plt.show()
