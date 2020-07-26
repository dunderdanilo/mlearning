import numpy as np 
from mlearning.models import LogisticRegression

clf = LogisticRegression(learning_rate=0.1)
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

clf.fit(X, y, verbose=True)

print(clf.params_)

X_test = np.arange(0,7).reshape((7,1))
y_hat_probs = clf.predict_probabilities(X_test)
y_hat = clf.predict(X_test)

print(y_hat_probs)
print(y_hat)
