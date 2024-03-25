import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class new_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(new_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # 入力ゲート
        self.Wii = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Whi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.zeros(hidden_size, 1))

        # 忘却ゲート
        self.Wif = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Whf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size, 1))

        # セルゲート
        self.Wig = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Whg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.zeros(hidden_size, 1))

        # 出力ゲート
        self.Wio = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Who = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.zeros(hidden_size, 1))

        # VIN考慮
        self.Wid = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Whd = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bd = nn.Parameter(torch.zeros(hidden_size, 1))

        #RNN
        self.Wro = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Who = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wrb = nn.Parameter(torch.zeros(hidden_size, 1))

    def forward(self, x, x_d, hidden, cell):
        h_prev = hidden 
        c_prev = cell 
        for t in range(self.hidden_size):
            # 入力ゲート 
            i_t = torch.sigmoid(self.Wii @ x + self.Whi @ h_prev[t, :].T + self.bi.T)
            # 忘却ゲート 
            f_t = torch.sigmoid(self.Wif @ x + self.Whf @ h_prev[t, :].T + self.bf.T)
            print(f_t.size)
            # セルゲート 
            g_t = torch.tanh(self.Wig @ x + self.Whg @ h_prev[t, :].T + self.bg.T)
            # 出力ゲート
            o_t = torch.sigmoid(self.Wio @ x + self.Who @ h_prev[t, :].T + self.bo.T)
            # VIX考慮
            d_t = torch.sigmoid(self.Wid @ x_d + self.Whd @ h_prev[t, :].T + self.bd.T)

            # 新しいセル状態
            # 新しいセル状態にVIX指数を考慮するd_tを加える。
            c_prev[t+1, :] = f_t * c_prev[t, :] + i_t * g_t + d_t
            # 新しい隠れ状態
            h_prev[t+1, :] = o_t @ torch.tanh(c_prev[t, :])
            if t == self.hidden_size-1:
                break

        output = torch.tanh(self.Wro @ x + self.Who @ h_prev.T + self.Wrb.T)

        return output, h_prev, c_prev
    
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size))
