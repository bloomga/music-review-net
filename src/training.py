from lstmModel import MusicLSTM
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

fname = "metacritic_reviews_test"
standardized = 0

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

#initialize model
#using basic hyperparameters for now, these will be optimized later
vocab_size = len(review_dict)+1 # +1 accounts for the 0 padding "word"
output_size = 1
input_size = 400
hidden_size = 256
num_rec_layers = 2
net = MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers)

#split datasets into train and test
train_x = reviews[:int(0.8*len(reviews))]
train_y = scores[:int(0.8*len(reviews))]
test_x = reviews[int(0.8*len(reviews)):]
test_y = scores[int(0.8*len(reviews)):]              

#create k-folds and loop
kfold =KFold(n_splits=10)
for fold, (train_index, test_index) in enumerate(kfold.split(train_x, train_y)):
    #divide up folds
    train_fold_x = train_x[train_index]
    test_fold_x = test_x[test_index]
    train_fold_y = train_y[train_index]
    test_fold_y = test_y[test_index]

    #create tensors and dataloaders
    batch_size=50
    train = torch.utils.data.TensorDataset(train_fold_x, train_fold_y)
    test = torch.utils.data.TensorDataset(test_fold_x, test_fold_y)
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

    
