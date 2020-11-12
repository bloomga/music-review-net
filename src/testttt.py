import json
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch

fname = "metacritic_reviews"
std_str = "Standardized"
with open("obj/" + fname + std_str +'Scores.json', "r") as fp:
    scores = json.load(fp)[:25]
    print(scores)
    def scale(scores):
        s_min = min(scores)
        s_max = max(scores)
        scaled = [( (((x-s_min)/(s_max-s_min))*(2)) -1) for x in scores]
        return scaled
    scores = scale(scores)
    print(scores)
with open("obj/encoded" + fname + 'Preprocessed.json', "r") as fp:
    reviews = json.load(fp)[:25]

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")

train_x = np.array(reviews)
train_y = np.array(scores)

batch_size = 25
train = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, drop_last = True)

for inputs, targets in train_loader:
    targets = targets.to(device)
    inputs = inputs.to(device).long()
