from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

X, y = load_iris(return_X_y=True)

data = np.array([
    [1,1],
    [1,1],
    [2,1],
    [2,0],
    [3,1],
    [3,0],
    [4,0],
    [4,0],
])

X = data[:, :-1]
y = data[:, -1]

clf = LogisticRegression(penalty='none').fit(X, y)

print(clf.coef_)

X_test = np.arange(0,7).reshape((7,1))

y_hat = clf.predict(X_test)
print(y_hat)

y_hat = clf.predict_proba(X_test)
print(y_hat)

# clf.predict(X[:2, :])
# # array([0, 0])

# clf.predict_proba(X[:2, :])
# # array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
# #        [9.7...e-01, 2.8...e-02, ...e-08]])
# >>> clf.score(X, y)
# 0.97...