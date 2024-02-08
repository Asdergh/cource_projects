import numpy as np
import matplotlib.pyplot as plt


X = np.random.normal(0.23, 12.34, (100, 34))
covar_matrix = np.cov(X)
print(X.shape, covar_matrix.shape)
plt.imshow(covar_matrix, cmap="magma")
plt.show()