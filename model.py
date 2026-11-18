import torch.nn as nn

'''
class HandRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_xh = nn.Parameter(torch.empty(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.W_y = nn.Parameter(torch.empty(hidden_size, output_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))

        nn.init.xavier_uniform_(self.W_xh)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.xavier_uniform_(self.W_y)

    def forward(self, x):
        batch_size = x.size(0)

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)


        for t in range(x.size(1)):
            x_t = x[:, t, :] 
            h = torch.tanh(
                x_t @ self.W_xh + h @ self.W_hh + self.b_h)

        pred = h @ self.W_y + self.b_y  
        return pred

'''
class mymodel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()
        self.loss_history = []
        self.epoch_record = []
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hn = self.rnn(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred
