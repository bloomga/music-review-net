from lstmModel import MusicLSTM
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import sys

#these will be changed to command line inputs
#fname = str(sys.argv[0])
fname = "metacritic_reviews"
standardized = 0 

#rmse loss function
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

#load reviews, scores, and encoding dict
std_str = "Preprocessed"
if standardized == 1:
    std_str = "Standardized"
with open("obj/" + fname + std_str +'Scores.json', "r") as fp:
    scores = json.load(fp)
with open("obj/encoded" + fname + 'Preprocessed.json', "r") as fp:
    reviews = json.load(fp)
with open("obj/" + fname + 'PreprocessedDict.json', "r") as fp:
    review_dict = json.load(fp)

print(len(scores[:2000]))
print(len(reviews[:2000]))
