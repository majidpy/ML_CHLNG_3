import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
X = [[0, 0, 3], [1, 1, 7], [0, 2, 1], [1, 0, 2]]
enc.fit(X)  


enc.n_values_

enc.feature_indices_

y = [[0, 1, 1]]
print(enc.transform(y).toarray())
