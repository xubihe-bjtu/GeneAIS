import torch.nn as nn
import torch


class ValueEmbedding(nn.Module):
    def __init__(self, c_in,d_model):
        super(ValueEmbedding, self).__init__()

        self.embed = nn.Conv2d(in_channels=c_in,out_channels=d_model,kernel_size=(1, 1))

    def forward(self, x):
        value_emb=self.embed(x)
        value_emb=value_emb
        return value_emb

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = ValueEmbedding(c_in=c_in,d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)


class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), dilation=(1, dilation),
                              padding=(0, (kernel_size - 1) * dilation // 2))
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size, max_dilation=16,d_model=64,d_mark=64):
        super(TCN, self).__init__()

        self.layers = nn.ModuleList()
        self.d_model=d_model
        self.d_mark=d_mark
        self.start_conv = DataEmbedding(in_channels, self.d_model, dropout=0.1)
        dilation = 1
        for i in range(num_layers):
            self.layers.append(TCNLayer(self.d_model+self.d_mark, out_channels, kernel_size, dilation))
            # 每层的膨胀因子加倍
            dilation = min(dilation * 2, max_dilation)

        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))  # 用于输出投影
        self.dropout=0.2



    def forward(self, x,x_mark):
        B,C,N,L = x.size()
        B,L,D=x_mark.shape
        x=self.start_conv(x)#B,D,N,L
        x_mark=x_mark.unsqueeze(2).repeat(1, 1, N, 1).permute(0,3,2,1)#B,D,N,L
        x=torch.cat([x,x_mark],dim=1)
        residual = x
        h=[]
        for layer in self.layers:
            x = layer(x)
            x = x + residual
            residual = x
            h.append(x)
        x = self.final_conv(x)
        return x,h