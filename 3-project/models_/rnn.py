import torch.nn as nn
import torch.nn.functional as f


class RNN(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(vocab.vocab.__len__(), args.d_model)
        self.rnn = nn.LSTM(args.d_model, args.d_model, batch_first=True, bidirectional=True,
                           num_layers=2, dropout=args.dropout)

        self.fc1 = nn.Linear(args.d_model * 2, args.d_model >> 2)
        self.fc2 = nn.Linear((args.d_model >> 2) * args.max_words, 3)

    def forward(self, x):

        x = self.emb(x)

        x = self.rnn(x)[0]
        x = f.dropout(f.relu(x), self.args.dropout, self.training)

        x = self.fc1(x)
        x = f.dropout(f.relu(x), self.args.dropout, self.training)

        x = x.view(x.size(0), -1)
        x = self.fc2(x)

        return x
