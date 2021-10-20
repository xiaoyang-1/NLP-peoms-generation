import torch
import numpy as np
from torch.utils.data import DataLoader

class TextConverter(object):
    def __init__(self, text, max_voc:int=5000):
        """
        文本字符数值化转换
        把字符转换为编号
        text: 需要转化的文本，string
        max_voc: 字符种类的最大值，int
        """
        voc_count = {} # 统计每个字符的频数
        for word in text:
            if word in voc_count:
                voc_count[word] += 1
            else:
                voc_count[word] = 1
                
        vocab = sorted(voc_count, key=voc_count.get, reverse=True)
        if len(vocab) > max_voc:
            # 如果训练集字符数超出了规定的数量，需要截取
            vocab = vocab[:max_voc]
        self.vocab = vocab
        
        # 字符和数值转换表
        self.word2intTab = {word: index for index, word in enumerate(vocab)}
        self.int2wordTab = {index: word for index, word in enumerate(vocab)}
        

    def vocab_size(self):
        """
        返回字典的大小，加1的原因是需要考虑到<unk>字符
        """
        return len(self.vocab) + 1
    
    def word2int(self, word):
        """
        字符转换为数值函数
        """
        if word in self.word2intTab:
            return self.word2intTab[word]
        else:
            # word不在字典里，当做unknown处理
            return len(self.vocab)
        
    def int2word(self, index):
        """
        数值转换为字符函数
        """
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int2wordTab[index]
        else:
            raise Exception("Unknown index!")
            
    def text2arr(self, text):
        """
        文本转换为数组函数
        """
        arr = []
        for word in text:
            arr.append(self.word2int(word))
        return np.array(arr)
    
    def arr2text(self, arr):
        """
        数组转换为文本函数
        """
        text = []
        for index in arr:
            text.append(self.int2word(index))
        return "".join(text)



class TextDataset():
    def __init__(self, arr):
        self.arr = arr
        
    def __getitem__(self, index):
        x = self.arr[index, :]
        y = torch.zeros(x.shape)
        
        y[:-1], y[-1] = x[1:], x[0]
        return x, y
    
    def __len__(self):
        return self.arr.shape[0]



def get_train_set():
    with open('./poetry.txt', 'r', encoding='utf-8') as f:
        poetry = f.read()
        poetry = poetry.replace('\n', '')
    
    convert = TextConverter(poetry, max_voc = 10000)
    # test_text = poetry[:12]

    num_word = 20 # 每个序列包含的字符数量
    num_seq = int(len(poetry) / num_word) # 序列的数量

    text = poetry[:num_seq * num_word]
    arr = convert.text2arr(text)
    arr = arr.reshape((num_seq, -1))

    arr = torch.from_numpy(arr)
    train_set = TextDataset(arr)

    return train_set, convert


if __name__ == '__main__':

    train_set, convert = get_train_set()
    train_data = DataLoader(train_set, 10)

    for data in train_data:
        x, y = data
        print(x[0])
        print(y[0])

        break
