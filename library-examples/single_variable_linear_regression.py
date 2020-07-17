import numpy as np
import matplotlib.pyplot as plt 
from mlearning.models import LinearRegression

clf = LinearRegression(learning_rate=0.1)
data = np.array([
    [1,1],
    [1,2],
    [2,2],
    [2,3],
    [3,3],
    [3,4],
    [4,4],
    [4,5],
])

X = data[:, :-1]
y = data[:, -1]

clf.fit(X, y)

print(clf.params_)

X_test = np.arange(0,7).reshape((7,1))
y_hat = clf.predict(X_test)

plt.scatter(X.reshape(X.shape[0]), y, c="red")
plt.plot(X_test, y_hat)
plt.show()

plt.plot(np.arange(1,len(self.loss_) + 1), self.loss_, c="red")
plt.show()
