import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # Initializing the hidden state to 0s
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size)
        # Making a pass through the RNN
        out, _ = self.rnn(X, h0)
        # Retrieving only the last sequence (=the most recent prediction)
        out = out[:, -1, :]
        # Making a pass through the fully connected layer
        out = self.fc(out)
        return self.sigmoid(out)
