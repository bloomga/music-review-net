import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

#epoch 2 has std dev/std err of
#fold 2 vs epoch
eps = [1,2,3,4,5,6,7,8,9]
mses = [0.059577, 0.054475, 0.053563, 0.051785, 0.051020, 0.050718, 0.050629, 0.050667, 0.050576]
rmses = [np.sqrt(x) for x in mses]
plt.plot(eps, mses)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.show()
plt.plot(eps, rmses, 'r')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.show()
