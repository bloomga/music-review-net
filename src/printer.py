import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

#epoch 2 has std dev/std err of
#fold 2 vs epoch
eps = [1,2,3,4,5,6,7,8,9]
#mses = [0.059577, 0.054475, 0.053563, 0.051785, 0.051020, 0.050718, 0.050629, 0.050667, 0.050576]
#rmses = [np.sqrt(x) for x in mses]
#plt.plot(eps, mses)
#plt.ylabel('MSE')
#plt.xlabel('Epoch')
#plt.show()
#plt.plot(eps, rmses, 'r')
#plt.ylabel('RMSE')
#plt.xlabel('Epoch')
#plt.show()

fname = 'metacritic_reviews'
with open("obj/" + fname + "TrainLosses.json", "r") as fp:
    train_losses = json.load(fp)
with open("obj/" + fname + "TrainRMSEs.json", "r") as fp:
    train_rmse = json.load(fp)
with open("obj/" + fname + "TrainBatchSteps.json", "r") as fp:
    train_steps = json.load(fp)
with open("obj/" + fname + "TrainEpochs.json", "r") as fp:
    train_epochs = json.load(fp)
with open("obj/" + fname + "TrainMinRes.json", "r") as fp:
    train_mins = json.load(fp)
with open("obj/" + fname + "TrainMaxRes.json", "r") as fp:
    train_maxs = json.load(fp)
with open("obj/" + fname + "ValMinRes.json", "r") as fp:
    val_mins = json.load(fp)
with open("obj/" + fname + "ValMaxRes.json", "r") as fp:
    val_maxs = json.load(fp)

print(train_epochs.index(2))
print(train_epochs.index(3))
print(train_epochs.index(4))
print(train_epochs.index(5))
print(train_epochs.index(6))
print(train_epochs.index(7))
print(train_epochs.index(8))
print(train_epochs.index(9))
print(len(train_steps))
print(len(train_epochs))

plt.plot(train_steps, train_losses)
plt.ylabel('MSE')
plt.xlabel('Batches')
plt.show()

plt.plot(train_steps, train_rmse)
plt.ylabel('RMSE')
plt.xlabel('Batches')
plt.show()

plt.plot(train_steps, train_maxs, 'b')
plt.plot(train_steps, train_mins, 'r')
plt.ylabel('Max/Min Residual')
plt.xlabel('Batches')
plt.show()

plt.plot(eps, val_maxs, 'b')
plt.plot(eps, val_mins, 'r')
plt.ylabel('Max/Min Residual')
plt.xlabel('Epochs')
plt.show()
