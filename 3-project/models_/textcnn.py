import torch
import torch.nn as nn
import torch.nn.functional as f


class TextCNN(nn.Module):
    def __init__(self, args, vocab):
        super(TextCNN, self).__init__()
        self.args = args

        filter_num = 128
        filter_sizes = [3, 4, 5]

        self.emb = nn.Embedding(vocab.vocab.__len__(), args.d_model)
        # self.rnn = nn.LSTM(args.d_model, args.d_model >> 1, batch_first=True, bidirectional=True)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, args.d_model)) for size in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(filter_sizes) * filter_num, 3)

    def forward(self, x):
        x = self.emb(x)
        # x = self.rnn(x)[0]
        # x = self.dropout(x)
        x = x.unsqueeze(1)
        x = [f.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [f.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
