from lstmModel import MusicLSTM
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import sys


#fname = str(sys.argv[1])
fname = "metacritic_reviews"
#if we want to use standardized dataset or not
standardized = False

def scale(scores):
    s_min = min(scores)
    s_max = max(scores)
    scaled = [( (((x-s_min)/(s_max-s_min))*(2)) -1) for x in scores]
    return scaled

#load reviews, scores, and encoding dict
std_str = "Preprocessed"
if standardized:
    std_str = "Standardized"
with open("obj/" + fname + std_str +'Scores.json', "r") as fp:
    scores = json.load(fp)
    scores = scale(scores)
with open("obj/encoded" + fname + 'Preprocessed.json', "r") as fp:
    reviews = json.load(fp)
with open("obj/" + fname + 'PreprocessedDict.json', "r") as fp:
    review_dict = json.load(fp)

#set hyperparameters
#using basic hyperparameters for now, these will be optimized later
vocab_size = len(review_dict)+1 # +1 accounts for the 0 padding "word"
output_size = 1
input_size = 400
hidden_size = 256
num_rec_layers = 2
dropout = 0.5
learning_rate = 0.001
batch_size = 10
epochs = 9
lin_layers = 2 #hackish but easier than reading number of layers from model


def residuals(output, targets):
    targets = targets.tolist()
    output = output.tolist()
    differences = []

    for i in range(len(output)):
        difference = output[i] - targets[i]
        differences.append(abs(difference))

    return max(differences), min(differences)



#loss function MSE
def MSE(yhat, y):
    return torch.mean((yhat-y)**2)

criterion = MSE

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")

train_x = np.array(reviews)
train_y = np.array(scores)

#create k-folds and loop
k = 5
kfold = KFold(n_splits=k)

final_val_losses = list()

for fold, (train_index, val_index) in enumerate(kfold.split(train_x, train_y)):
    if(fold + 1 == 2):
        #initialize model
        net = MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout)

        #optmizer
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        #put into training mode
        net.train()

        #divide up folds
        train_fold_x = train_x[train_index]
        val_fold_x = train_x[val_index]
        train_fold_y = train_y[train_index]
        val_fold_y = train_y[val_index]

        print("Fold: {}/{}...".format(fold+1, k),
            "Validation set Std Dev.: {:.6f}...".format(np.std(val_fold_y)))

        #create tensors and dataloaders
        train = TensorDataset(torch.FloatTensor(train_fold_x), torch.FloatTensor(train_fold_y))
        validate = TensorDataset(torch.FloatTensor(val_fold_x), torch.FloatTensor(val_fold_y))
        train_loader = DataLoader(train, batch_size = batch_size, shuffle = True, drop_last = True)
        val_loader = DataLoader(validate, batch_size = batch_size, shuffle = True, drop_last = True)

        step_counter = 0

        printed_train_losses = list()
        printed_train_maxes = list()
        printed_train_mins = list()
        printed_train_rmses = list()
        printed_train_epochs = list()
        printed_train_steps = list()
        printed_val_losses = list()
        printed_val_maxes = list()
        printed_val_mins = list()
        
        
        #epoch loop
        for e in range(epochs):
            print("Epoch: {}/{}...".format(e+1, epochs))
            #init zeroed hidden state
            hidden = net.init_hidden_state(batch_size, train_on_gpu)

            #batch loop
            for inputs, targets in train_loader:
                step_counter += 1

                inputs = inputs.to(device).long()
                targets = targets.to(device)

                #create new hidden state variables
                hidden = tuple([h.data for h in hidden])

                #zero out gradients
                net.zero_grad()

                #get output of music lstm
                output, hidden = net(inputs, hidden)
                output = output.view(batch_size, -1)[:,-1]

                #calculate loss and backwards propogate
                loss = criterion(output, targets)
                loss.backward()

                #built-in function to help prevent the exploding gradient problem that is common in RNN's
                nn.utils.clip_grad_norm_(net.parameters(), 5)

                #update parameters
                optimizer.step()



                #calculate loss stats
                if step_counter % 25 == 0: #currently lower print rate for testing (turn off for grid search)
                    rmse = np.sqrt(loss.item())
                    maxResidual, minResidual = residuals(output, targets)
                    print("Fold: {}/{}...".format(fold+1, k),
                          "Epoch: {}/{}...".format(e+1, epochs),
                          "Step: {}...".format(step_counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "RMSE: {:.6f}...".format(rmse),
                          "Max Residual: {:.6f}...".format(maxResidual),
                          "Min Residual: {:.6f}...".format(minResidual))

                    if(fold + 1 == 2):
                        printed_train_losses.append(loss.item())
                        printed_train_rmses.append(rmse)
                        printed_train_maxes.append(maxResidual)
                        printed_train_mins.append(minResidual)
                        printed_train_steps.append(step_counter)
                        printed_train_epochs.append(epoch)


            #calculate validation loss
            #first init a new zeroed hidden state
            val_hidden = net.init_hidden_state(batch_size, train_on_gpu)
            val_losses = list()
            maxFinal = 0
            minFinal = 10

            with torch.no_grad():
                net.eval() #put net in eval mode so it doesnt learn from the validation data
                for inputs, targets in val_loader:

                    inputs = inputs.to(device).long()
                    targets = targets.to(device)

                    #create new hidden state variables
                    val_hidden = tuple([h.data for h in val_hidden])

                    #get output and then calculate loss
                    output, val_hidden = net(inputs, val_hidden)
                    output = output.view(batch_size, -1)[:,-1]
                    val_loss = criterion(output, targets)

                    val_losses.append(val_loss.item())

                    maxResidualVal, minResidualVal = residuals(output, targets)

                    if minFinal > minResidualVal:
                        minFinal = minResidualVal

                    elif maxFinal< maxResidualVal:
                        maxFinal = maxResidualVal
                    

                val_loss = np.mean(val_losses)
                val_rmse = np.sqrt(val_loss)
                net.train() #set back to training mode

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Val Loss: {:.6f}...".format(val_loss),
                      "Val RMSE: {}...".format(val_rmse),
                      "Max Val Residual: {:.6f}...".format(maxFinal),
                      "Min Val Residual: {:.6f}...".format(minFinal))

                if(fold + 1 == 2):
                    printed_val_losses.append(val_loss)
                    printed_val_maxes.append(maxFinal)
                    printed_val_mins.append(minFinal)

        #dump val losses, val rmses, and residues for "zoom in" on fold       
        if(fold + 1 == 2):
            printed_val_rmses = [sqrt(x) for x in printed_val_losses]
            with open("obj/" + fname + "ValLosses.json", "w") as fp:
                json.dump(printed_val_losses, fp)
            with open("obj/" + fname + "ValRMSEs.json", "w") as fp:
                json.dump(printed_val_losses, fp)
            with open("obj/" + fname + "ValMaxRes.json", "w") as fp:
                json.dump(printed_val_maxes, fp)
            with open("obj/" + fname + "ValMinRes.json", "w") as fp:
                json.dump(printed_val_mins, fp)
            with open("obj/" + fname + "TrainLosses.json", "w") as fp:
                json.dump(printed_train_losses, fp)
            with open("obj/" + fname + "TrainRMSEs.json", "w") as fp:
                json.dump(printed_train_losses, fp)
            with open("obj/" + fname + "TrainMaxRes.json", "w") as fp:
                json.dump(printed_train_maxes, fp)
            with open("obj/" + fname + "TrainMinRes.json", "w") as fp:
                json.dump(printed_train_mins, fp)
            with open("obj/" + fname + "TrainBatchSteps.json", "w") as fp:
                json.dump(printed_train_steps, fp)
            with open("obj/" + fname + "TrainEpochs.json", "w") as fp:
                json.dump(printed_train_epochs, fp)
                
        final_val_losses.append(val_loss)

#print out final validation stats (averages over cross validation)
print("Final validation stats after cross validation is done")
print("Learning Rate: {:.6f}...".format(learning_rate))
print("Number of Epochs: {:.6f}...".format(epochs))
print("Batch size: {:.6f}...".format(batch_size))
print("Number of LSTM Layers: {:.6f}...".format(num_rec_layers))
print("Number of Linear/Dense Layers: {:.6f}...".format(lin_layers))
print("Val Loss: {:.6f}...".format(np.mean(final_val_losses)))
print("Val RMSE: {:.6f}...".format(np.sqrt(np.mean(final_val_losses))))
print("Standard Error: {:.6f}".format((np.std(final_val_losses))/(np.sqrt(k))))


