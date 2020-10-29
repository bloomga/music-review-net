from lstmModel import MusicLSTM
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import sys

#these will be changed to command line inputs
#fname = str(sys.argv[1])
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
    scores = json.load(fp)[:1000] #for testing
with open("obj/encoded" + fname + 'Preprocessed.json', "r") as fp:
    reviews = json.load(fp)[:1000] #for testing
with open("obj/" + fname + 'PreprocessedDict.json', "r") as fp:
    review_dict = json.load(fp)

#initialize model
#using basic hyperparameters for now, these will be optimized later
vocab_size = len(review_dict)+1 # +1 accounts for the 0 padding "word"
output_size = 1
input_size = 400
hidden_size = 256
num_rec_layers = 2
dropout = 0.5
net = MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout)
print(net)

#loss function RMSE
criterion = RMSELoss
    
#optmizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")

#put into training mode
net.train()

#split datasets into train and test
train_x = np.array(reviews[:int(0.8*len(reviews))])
train_y = np.array(scores[:int(0.8*len(reviews))])
reserved_test_x = np.array(reviews[int(0.8*len(reviews)):])
reserved_test_y = np.array(scores[int(0.8*len(reviews)):])             

#create k-folds=10 and loop
kfold = KFold(n_splits=4) #we can lower the splits number for testing
for fold, (train_index, test_index) in enumerate(kfold.split(train_x, train_y)):
    #divide up folds
    train_fold_x = train_x[train_index]
    test_fold_x = train_x[test_index]
    train_fold_y = train_y[train_index]
    test_fold_y = train_y[test_index]

    #create tensors and dataloaders
    batch_size=50
    train = TensorDataset(torch.FloatTensor(train_fold_x), torch.FloatTensor(train_fold_y))
    test = TensorDataset(torch.FloatTensor(test_fold_x), torch.FloatTensor(test_fold_y))
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test, batch_size = batch_size, shuffle = True)


    epochs = 4 #we will adjust this after training, see how many epochs before loss stops decreasing
    step_counter = 0
    
    #epoch loop
    for e in range(epochs):
        #init zeroed hidden state
        hidden = net.init_hidden_state(batch_size, train_on_gpu)

        #batch loop
        for inputs, targets in train_loader:
            step_counter += 1
            
            inputs = inputs.to(device).long()
            targets = targets.to(device).long()
            
            #create new hidden state variables
            hidden = tuple([h.data for h in hidden])

            #zero out gradients
            net.zero_grad()

            #get output of music lstm
            output, hidden = net(inputs, hidden)

            #calculate loss and backwards propogate
            loss = criterion(output, targets)
            loss.backward()

            #built-in function to help prevent the exploding gradient problem that is common in RNN's
            nn.utils.clip_grad_norm_(net.parameters(), 5)

            #update parameters
            optimizer.step()

            #calculate loss stats
            #WE WILL INCLUDE MORE DATA GATHERING IN THIS STEP TOO
            if step_counter % 50 == 0: #change the print rate for testing
                #calculate validation loss
                #first init a new zeroed hidden state
                val_hidden = net.init_hidden_state(batch_size, train_on_gpu)
                val_losses = list()

                net.eval() #put net in eval mode so it doesnt learn from the validation data
                for inputs, targets in test_loader:

                    inputs = inputs.to(device).long()
                    targets = targets.to(device).long()

                    #create new hidden state variables
                    val_hidden = tuple([h.data for h in val_hidden])

                    #get output and then calculate loss
                    output, val_hidden = net(inputs, val_hidden)
                    val_loss = criterion(output, targets)

                    val_losses.append(val_loss.item())

                val_loss = np.mean(val_losses)
                net.train() #set back to training mode

                #CHANGE FROM PRINTING TO STORAGE FOR LATER PLOTTING
                #CALL PLOTTING FROM SEPERATE IMPORTED FILE, DONT CROWD THIS ONE
                print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(val_loss))

