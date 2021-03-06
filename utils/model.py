import torch
import torch.nn as nn
input_size = 16   # rnn input size
output_size = 5
lr = 0.002
batch_size = 1
num_epochs = 201
seq_length = 365
threshold = 2


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=128,  # rnn hidden unit
            num_layers=20,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        # else:
        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        out = torch.stack(outs, dim=1)
        out = self.relu(out)

        return out, h_state