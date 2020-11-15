import lstmModel3layers as lstmModel3
import lstmModel1layers as lstmModel1
import lstmModel2layers as lstmModel2
import lstmModel4layers as lstmModel4
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import sys

#CODE make this all into a function that accepts hyper-parameter settings 
#and loops over all options. put in new file called grid search.


#fname = str(sys.argv[1])
fname = "metacritic_reviews"
#if we want to use standardized dataset
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

print("dataset std deviation: " + str(np.std(scores)))
print("dataset mean: " + str(np.mean(scores)))

def train(num_lin_layers, rec_layers, learn_rate, batch, eps):

    #set hyperparameters
    #using basic hyperparameters for now, these will be optimized later
    vocab_size = len(review_dict)+1 # +1 accounts for the 0 padding "word"
    output_size = 1
    input_size = 400
    hidden_size = 256
    num_rec_layers = rec_layers
    dropout = 0.5
    learning_rate = learn_rate
    lin_layers = num_lin_layers
    batch_size = 25
    epochs = eps 
    
    print("Learning Rate: {:.6f}...".format(learning_rate))
    print("Number of Epochs: {:.6f}...".format(epochs))
    print("Batch size: {:.6f}...".format(batch_size))
    print("Number of LSTM Layers: {:.6f}...".format(num_rec_layers))
    print("Number of Linear/Dense Layers: {:.6f}...".format(lin_layers))

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

    final_val_r2s = list()
    final_val_losses = list()
    
    for fold, (train_index, val_index) in enumerate(kfold.split(train_x, train_y)):
    
        #initialize model
        if lin_layers == 2:
            net = lstmModel2.MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout)
        if lin_layers == 3:
            net = lstmModel3.MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout)
        if lin_layers == 4:
            net = lstmModel4.MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout)
        else:
            net = lstmModel1.MusicLSTM(vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout)

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
                if(False and step_counter % 20 == 0): #turned off training prining for gridsearch
                    r2 = r2_score(targets.tolist(), output.tolist())
                    rmse = np.sqrt(loss.item())
                    print("Fold: {}/{}...".format(fold+1, k), 
                          "Epoch: {}/{}...".format(e+1, epochs),
                          "Step: {}...".format(step_counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "R^2: {}...".format(r2),
                          "RMSE: {}...".format(rmse))

                    
            #calculate validation loss
            #first init a new zeroed hidden state
            val_hidden = net.init_hidden_state(batch_size, train_on_gpu)
            val_losses = list()
            val_r2s = list()

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

                    val_r2 = r2_score(targets.tolist(), output.tolist())
                    val_r2s.append(val_r2)

                val_loss = np.mean(val_losses)
                val_r2 = np.mean(val_r2s)
                val_rmse = np.sqrt(val_loss)
                net.train() #set back to training mode
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Val Loss: {:.6f}...".format(val_loss),
                      "Val R^2: {}...".format(val_r2),
                      "Val RMSE: {}...".format(val_rmse))
        
        final_val_losses.append(val_loss)
        final_val_r2s.append(val_r2)

    #print out final validation stats (averages over cross validation)
    print("Final validation stats after cross validation is done")
    print("Learning Rate: {:.6f}...".format(learning_rate))
    print("Number of Epochs: {:.6f}...".format(epochs))
    print("Batch size: {:.6f}...".format(batch_size))
    print("Number of LSTM Layers: {:.6f}...".format(num_rec_layers))
    print("Number of Linear/Dense Layers: {:.6f}...".format(lin_layers))
    print("Val Loss: {:.6f}...".format(np.mean(final_val_losses)))
    print("Val R^2: {:.6f}...".format(np.mean(final_val_r2s)))
    print("Val RMSE: {:.6f}...".format(np.sqrt(np.mean(final_val_losses))))
    print("Standard Error: {:.6f}".format((np.std(final_val_losses))/(np.sqrt(k))))


ep_list = [3, 9, 27]
lr_list = [0.0001, 0.0005, 0.001, 0.005]
batch_list = [10, 30, 50, 70, 90]
lin_layer_list = [2]
lstm_layer_list = [2]
#gridsearching
for eps in ep_list:
    for learn_rate in lr_list:
        for batch in batch_list:
            for num_lin_layers in lin_layer_list:
                for rec_layers in lstm_layer_list:
                    train(num_lin_layers, rec_layers, learn_rate, batch, eps)
