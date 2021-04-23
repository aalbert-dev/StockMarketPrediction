# A very basic example of building a model using basic LSTM layer
import torch
import torch.nn as nn

INPUT_SIZE = 64
H_SIZE = 128
LAYERS = 3
OUTPUT_SIZE = 1

class LSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=H_SIZE, num_layers=LAYERS, output_size=OUTPUT_SIZE):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,    # lstm hidden unit
            num_layers=num_layers,      # number of lstm layer
            batch_first=True,           # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        # self.dropout = nn.Dropout(p=0.05) 		    # Dropout layer, optional. May or may not be useful based on implementation

        self.out = nn.Linear(hidden_size, output_size)  # Output layer, Linear transformation is commonly used

        # self.embedding = nn.Linear(128, 64)           # Linear transformation could also do embedding, optional

        self.relu = nn.ReLU() 			 # ReLU activiation, if want non-negative outputs
        self.softmax = nn.Softmax(dim=1) # Softmax activation, best for output a distribution of probability
        self.sigmoid = nn.Sigmoid() 	 # Sigmoid activation, map output value into (0, 1) interval

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size), pass into model when training/predicting
        # r_out (batch, time_step, hidden_size)
        
        r_out, h_state = self.lstm(x, h_state)
        outs = []                                 # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        out = torch.stack(outs, dim=1)
        # out = self.dropout(out) # Optional dropout

        # Activiation layer is optional. Depends on your implementation
        out = self.relu(out)	  # Use correct activation function for output, i.e., scalar value or classification?

        return out, h_state