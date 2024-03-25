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

        # 考慮
        self.Wid = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Whd = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bd = nn.Parameter(torch.zeros(hidden_size, 1))

        #RNN
        self.Wro = nn.Parameter(torch.randn(hidden_size, input_size))
        self.Who = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wrb = nn.Parameter(torch.zeros(hidden_size, 1))

    def forward(self, x, x_d, hidden, cell):
        h_prev = hidden #(40, 512)
        c_prev = cell #(40, 512)
        for t in range(self.hidden_size):
            # 入力ゲート 
            #(512, 40) @ (40, 1) + (512, 512) @ (512, 1) + (512, 1) = (1, 512)
            i_t = torch.sigmoid(self.Wii @ x + self.Whi @ h_prev[t, :].T + self.bi.T)
            # 忘却ゲート (1, hidden_size)
            f_t = torch.sigmoid(self.Wif @ x + self.Whf @ h_prev[t, :].T + self.bf.T)
            print(f_t.size)
            # セルゲート (1, hidden_size)
            g_t = torch.tanh(self.Wig @ x + self.Whg @ h_prev[t, :].T + self.bg.T)
            # 出力ゲート (1, hidden_size) 
            o_t = torch.sigmoid(self.Wio @ x + self.Who @ h_prev[t, :].T + self.bo.T)
            # 考慮 (1, hidden_size)
            d_t = torch.sigmoid(self.Wid @ x_d + self.Whd @ h_prev[t, :].T + self.bd.T)

            # 新しいセル状態
            # (1, 512) @ (1, 512) + (1, hidden_size) @ (1, hidden_size) + (1, hidden_size)

            c_prev[t+1, :] = f_t * c_prev[t, :] + i_t * g_t + d_t
            # 新しい隠れ状態
            # (1, 512) @ (1, 512)
            h_prev[t+1, :] = o_t @ torch.tanh(c_prev[t, :])
        output = torch.tanh(self.Wro @ x + self.Who @ h_prev.T + self.Wrb.T)
        return output, h_prev, c_prev
    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size),
                torch.zeros(batch_size, self.hidden_size))

# # パラメータ
# input_size = 10
# hidden_size = 20
# batch_size = 1

# # モデルのインスタンス化
# model = new_LSTM(input_size, hidden_size)

# # ダミーデータと初期隠れ状態の準備
# x_dummy = torch.randn(batch_size, input_size)
# x_d = torch.randn(batch_size, input_size)
# hidden = model.init_hidden(batch_size)

# # モデルの実行
# h_next, c_next = model(x_dummy,x_d, hidden)

# print("Next hidden state:", h_next)
# print("Next cell state:", c_next)
