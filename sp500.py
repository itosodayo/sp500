import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model_3 import new_LSTM
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from datetime import datetime


data_sp500 = pd.read_csv('RNNmodel/sp500.csv', header=0)
data_sp500 = data_sp500['Close'] - data_sp500['Open']


seq_length = 40
chunk_size = seq_length+1
sp500_chunks = [data_sp500.iloc[i:i+chunk_size] for i in range(len(data_sp500)-chunk_size+1)]


sp500_chunks = torch.from_numpy(np.array(sp500_chunks)).float()


class dataset(Dataset):
    def __init__(self, data_chunks):
        self.data_chunks = data_chunks

    def __len__(self):
        return len(self.data_chunks)
    
    def __getitem__(self, idx):
        data_chunk = self.data_chunks[idx]
        return data_chunk[:-1].long(), data_chunk[1:].long()

    
sp500_dataset = dataset(torch.tensor(sp500_chunks))
batch_size = 64
seq_dl = DataLoader(sp500_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class Model(nn.Module):
    def __init__(self, input_size, rnn_hidden_size):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(input_size, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, input_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        out, (hidden, cell) = self.rnn(x, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.rnn_hidden_size)
        cell = torch.zeros(batch_size, self.rnn_hidden_size)
        return hidden, cell

rnn_hidden_size = 512
input_size = seq_length
model = Model(input_size, rnn_hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epoch = 1000
batch_size = 100

sp500_dl = DataLoader(sp500_dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epoch):
    hidden, cell = model.init_hidden(batch_size)
    seq_batch, target_batch = next(iter(seq_dl))
    optimizer.zero_grad()
    loss = 0
    for c in range(seq_length):
        pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
        loss += loss_fn(pred, target_batch[:, c])

    loss.backward()
    optimizer.step()
    loss = loss.item()/seq_length
    if epoch % 500==0:
        print(epoch, loss)

 