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
data_VIX = pd.read_csv('RNNmodel/VIX.csv', header=0)

data_sp500['Date'] = pd.to_datetime(data_sp500['Date'])
data_sp500['Date'] = data_sp500['Date'].dt.strftime("%m/%d/%Y")
data_sp500 = data_sp500[data_sp500['Date'].isin(data_VIX['DATE'])]
data_VIX = data_VIX[data_VIX['DATE'].isin(data_sp500['Date'])]

data_sp500 = data_sp500['Close'] - data_sp500['Open']
data_VIX = (data_VIX['CLOSE'] + data_VIX['OPEN'])/2


seq_length = 40
chunk_size = seq_length+1
sp500_chunks = [data_sp500.iloc[i:i+chunk_size] for i in range(len(data_sp500)-chunk_size+1)]
VIX_chunks = [data_VIX.iloc[i:i+chunk_size] for i in range(len(data_VIX)-chunk_size+1)]


sp500_chunks = torch.from_numpy(np.array(sp500_chunks)).float()
VIX_chunks = torch.from_numpy(np.array(VIX_chunks)).float()

class dataset(Dataset):
    def __init__(self, data_chunks):
        self.data_chunks = data_chunks

    def __len__(self):
        return len(self.data_chunks)
    
    def __getitem__(self, idx):
        data_chunk = self.data_chunks[idx]
        return data_chunk[:-1], data_chunk[1:]

sp500_dataset = dataset(sp500_chunks)
VIX_dataset = dataset(VIX_chunks)

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn_hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, input_size)
        self.model = new_LSTM(input_size, hidden_size)

    def forward(self, x, x_d,  hidden, cell):
        out, hidden, cell = self.model(x.unsqueeze(1), x_d, hidden, cell)
        out = self.fc(out)
        return out, hidden, cell
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden, cell
    
rnn_hidden_size = 512
input_dim = seq_length
model = MyModel(input_dim, rnn_hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epoch = 1000
batch_size = 100

sp500_dl = DataLoader(sp500_dataset, batch_size=batch_size, shuffle=True)
VIX_chunks = DataLoader(VIX_dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epoch):
    hidden, cell = model.init_hidden(batch_size)
    sp500_seq_batch, sp500_target_batch = next(iter(sp500_dl))
    VIX_seq_batch, _ = next(iter(VIX_chunks))
    optimizer.zero_grad()
    loss = 0
    outputs = []
    print(sp500_target_batch.shape)
    for c in range(seq_length):
        pred, hidden, cell = model(sp500_seq_batch[c, :], VIX_seq_batch[c, :], hidden, cell)  #sp500_seq_batch=(100, 40) #hidden = (100, 512)
        outputs.append(pred)
        loss += loss_fn(outputs, sp500_target_batch[c, :])
    # loss.backward()
    # optimizer.step()
    # loss = loss.item()/seq_length
    # if epoch % 500==0:
    #     print(epoch, loss)