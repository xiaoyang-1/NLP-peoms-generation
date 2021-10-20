import torch

from torch.utils.data import DataLoader
from dataset import *
from model import *

def pick_top_n(pred, top_n=5):
    top_prob, top_label = torch.topk(pred, top_n, 1)
    top_prob /= torch.sum(top_prob)
    top_prob = top_prob.squeeze(0).cpu().numpy()
    top_label = top_label.squeeze(0).cpu().numpy()
    return np.random.choice(top_label, size=1, p=top_prob)


if __name__ == '__main__':

    begin = "谁还不是个小宝贝，"
    text_len = 100

    train_set, convert = get_train_set()
    model = CharRNN(convert.vocab_size(), 100, 100, 1, 0.5).to(device)
    state_dict = torch.load('CharRNN2.pth')
    model.load_state_dict(state_dict)

    samples = [index for index in convert.text2arr(begin)]
    input_txt = torch.LongTensor(samples)[None].to(device)
    _, init_state = model(input_txt)
    model_input = input_txt[:, -1][:, None].to(device)

    result = samples

    for i in range(text_len):
        out, init_state = model(model_input, init_state)
        pred = pick_top_n(out.data)
        model_input = torch.LongTensor(pred)[None].to(device)
        result.append(pred[0])

    text = convert.arr2text(result)
    print(text)