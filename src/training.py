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
    scores = json.load(fp)
with open("obj/encoded" + fname + 'Preprocessed.json', "r") as fp:
    reviews = json.load(fp)
with open("obj/" + fname + 'PreprocessedDict.json', "r") as fp:
    review_dict = json.load(fp)

print("dataset std deviation: " + str(np.std(scores)))
print("dataset mean: " + str(np.mean(scores)))

#set hyperparameters
#using basic hyperparameters for now, these will be optimized later
vocab_size = len(review_dict)+1 # +1 accounts for the 0 padding "word"
output_size = 1
input_size = 400
hidden_size = 256
num_rec_layers = 2
dropout = 0.5

#loss function RMSE
criterion = RMSELoss

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")

#split datasets into train and test
train_x = np.array(reviews[:int(0.9*len(reviews))])
train_y = np.array(scores[:int(0.9*len(reviews))])
reserved_test_x = np.array(reviews[int(0.9*len(reviews)):])
reserved_test_y = np.array(scores[int(0.9*len(reviews)):])             

#create k-folds and loop
k = 10 #want k=10. we can change k for testing
kfold = KFold(n_splits=k) 
model_list = list()
val_loss_list = list()

for fold, (train_index, val_index) in enumerate(kfold.split(train_x, train_y)):
    #initialize model
    net = MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout)

    #optmizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    #put into training mode
    net.train() 
    
    #divide up folds
    train_fold_x = train_x[train_index]
    val_fold_x = train_x[val_index]
    train_fold_y = train_y[train_index]
    val_fold_y = train_y[val_index]

    print(len(train_fold_x))
    print(len(val_fold_x))   
    #create tensors and dataloaders
    batch_size = 25
    train = TensorDataset(torch.FloatTensor(train_fold_x), torch.FloatTensor(train_fold_y))
    validate = TensorDataset(torch.FloatTensor(val_fold_x), torch.FloatTensor(val_fold_y))
    train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(validate, batch_size = batch_size, shuffle = True, drop_last = True)


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
            if step_counter % 10 == 0: #change the print rate for testing
                #CODE find training r^2, similar method to loss
                #CODE find largest and smallest (absolute value) residuals from all outputs
                #vs targets
                #CODE save these all in lists for each fold along with epoch/step/and loss
                print("Fold: {}/{}...".format(fold+1, k), 
                      "Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(step_counter),
                      "Loss: {:.6f}...".format(loss.item()))
                  
        #calculate validation loss
        #first init a new zeroed hidden state
        val_hidden = net.init_hidden_state(batch_size, train_on_gpu)
        val_losses = list()

        net.eval() #put net in eval mode so it doesnt learn from the validation data
        for inputs, targets in val_loader:

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
        #CODE find val r^2, similar method to val_loss
        #CODE find largest and smallest (absolute value) residual from all outputs vs targets
        #CODE store these and val_losses all in a list
        
        print("Epoch: {}/{}...".format(e+1, epochs),
              "Val Loss: {:.6f}...".format(val_loss))

    model_list.append(net)
    val_loss_list.append(val_loss)
    #CODE store Val Loss, val r^2, and min/max residuals vs Epoch for each fold
    #and its accompanying model in seperate lists
    
#determine best model
lowest_val_loss = 100000
index = 0
for i in range(len(model_list)):
    if val_loss_list[i] <= lowest_val_loss:
        lowest_val_loss = val_loss_list[i]
        index = i

#test the best model to get statistics
net = model_list[i]

#optmizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

#create new hidden state variables
test_hidden = net.init_hidden_state(batch_size, train_on_gpu)

#put into eval mode
net.eval()

test = TensorDataset(torch.FloatTensor(reserved_test_x), torch.FloatTensor(reserved_test_y))
test_loader = DataLoader(test, batch_size = batch_size, shuffle = True, drop_last = True)

test_losses = list()

for inputs, targets in test_loader:
    
    inputs = inputs.to(device).long()
    targets = targets.to(device).long()

    #create new hidden state variables
    test_hidden = tuple([h.data for h in test_hidden])

    #get output and then calculate loss
    output, test_hidden = net(inputs, test_hidden)
    test_loss = criterion(output, targets)
    test_losses.append(test_loss.item())
    #CODE find test r^2, similar method to test_loss
    #CODE find largest and smallest (absolute value) residual from all outputs vs targets

#CODE graph scatter plot of all outputs concacenated together
#and all targets concacenated together
test_loss=np.mean(test_losses)
print("Test Loss: {:.6f}".format(test_loss))
#CODE graph Val Loss, val r^2, and min/max residuals vs Epoch for each fold
#and its accompanying model
#CODE graph training Loss, training r^2, and min/max residuals vs step (annotated by epoch)
#for each fold and its accompanying model
