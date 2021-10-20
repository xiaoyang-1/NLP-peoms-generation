import torch

from torch.utils.data import DataLoader
from dataset import *
from model import *

if __name__ == '__main__':
    # hyper-parameter
    learning_rate = 0.001
    batch_size = 128
    epoches = 10
    # 训练数据读取

    train_set, convert = get_train_set()
    train_data = DataLoader(train_set, batch_size)
    
    # 模型初始化
    model = CharRNN(convert.vocab_size(), 100, 100, 1, 0.5).to(device)
    # 利用交叉熵作为loss function
    criterion = nn.CrossEntropyLoss()
    # 利用Adam算法作为模型优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for e in range(epoches):
        train_loss = 0
        for data in train_data:
            x, y = data
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            
            # 前向传播
            output, _ = model(x)
            loss = criterion(output, y.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 因为RNN存在梯度爆炸的问题，所以要进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            train_loss += loss.item()
            print(loss.item(), end="\r")

        print(f"Epoch: {e+1}/{epoches}, Loss: {train_loss/batch_size:.2f}")
        torch.save(model.state_dict(), "CharRNN2.pth")