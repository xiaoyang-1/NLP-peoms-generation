import torch
import torch.nn as nn

device = torch.device('cpu')

class CharRNN(nn.Module):
    def __init__(self, char_classes, embed_dim, hidden_size, rnn_layers, dropout):
        """
        Input:
            char_classes: 字典里字符的种类数，len(vocab) + 1
            embed_dim: 词向量维数
            hidden_size: RNN hidden_state 维数
            rnn_layers: RNN的层数
            dropout: RNN的dropout概率
        """
        super(CharRNN, self).__init__()
        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        
        # 词嵌入层
        self.embedding = nn.Embedding(char_classes, embed_dim)
        # RNN GRU 层
        self.rnn = nn.GRU(embed_dim, hidden_size, rnn_layers, dropout=dropout)
        # 线性映设层
        self.project = nn.Linear(hidden_size, char_classes)
        
    def forward(self, x, hs=None):
        """
        前向传播函数
        Input:
            x: (batch, len_of_seq), 包含有batch个序列
            hs: (rnn_layers, batch, hidden_size), hidden states of each layer
        """
        batch = x.shape[0]
        if hs is None:
            hs = torch.zeros(self.rnn_layers, batch, self.hidden_size).to(device)
        x = self.embedding(x) # (batch, len_of_seq, embed_dim)
        # 因为再Pytorch的GRU中要求输入的数据的结构为 (len_of_seq, batch, -1)
        # 则需要改变x的结构
        x = x.permute(1, 0, 2)
        # 输入到RNN中
        # RNN会有两个输出，一个是对应的序列output，另一个是hidden state
        out, h0 = self.rnn(x, hs)
        le, ba, hd = out.shape
        # 为了输入到映射层，我们要有改变out的结构
        out = out.view(le*ba, hd)
        out = self.project(out)
        # 然后将out的结构变回来
        out = out.view(le, ba, -1)
        # 再变成(ba, le, -1)
        out = out.permute(1, 0, 2).contiguous()
        return out.view(-1, out.shape[2]), h0

