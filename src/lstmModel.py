import torch.nn as nn

class MusicLSTM(nn.Module):
    
    #with starting point hyperparameters
    #set up the layers of the lstm model
    def __init__(self, vocab_size, output_size, input_size, hidden_size, num_rec_layers, dropout):

        super().__init__()
        self.output_dim = output_size
        self.num_rec_layers = num_rec_layers
        self.hidden_size = hidden_size
        
        #embedding layer
        self.embedding = nn.Embedding(vocab_size, input_size)

        #lstm layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_rec_layers, dropout=dropout, batch_first=True)

        #dropout layer
        self.dropout = nn.Dropout(0.3)

        #fully connected linear layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, output_size)

        #final ReLU layer (we dont need values less than 0)
        self.relu = nn.ReLU()


    #feeds forward some input x and hidden state through the model to produce an output
    def forward(self, x, hidden):

        #embedding output
        embedded = self.embedding(x)

        #lstm output
        lstm_out, hidden_state = self.lstm(embedded, hidden)

        #pytorch expects a contiguous tensor
        lstm_out_contig = lstm_out.contiguous().view(-1, self.hidden_size)

        #dropout
        out = self.dropout(lstm_out)

        #fully connected layers
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)

        #relu layer
        out = self.relu(out)

        return out, hidden


    #initializes tensors for the hidden state and the cell state of the lstm
    def init_hidden_state(self, batch_size, train_on_gpu):
        weight = next(self.parameters()).data

        if(train_on_gpu):
            hidden = (weight.new(self.num_rec_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.num_rec_layers, batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.num_rec_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_rec_layers, batch_size, self.hidden_size).zero_())
        return hidden

    
