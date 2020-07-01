import torch
import torch.nn as nn
import torch.nn.functional as f


class Attention(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(vocab.vocab.__len__(), args.d_model)
        self.rnn = nn.LSTM(args.d_model, int(args.d_model / 10), batch_first=True,
                           bidirectional=True)
        self.t_q = nn.Linear(args.d_model, args.d_model)
        self.t_k = nn.Linear(args.d_model, args.d_model)
        self.t_v = nn.Linear(args.d_model, args.d_model)
        self.fc = nn.Linear(int(args.d_model / 5) * args.max_words, 3)

    def forward(self, x):
        x = self.emb(x)
        q = self.t_q(x)
        k = self.t_k(x)
        v = self.t_v(x)
        w = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        w = f.softmax(w, dim=2)
        x = torch.matmul(w, v)
        x = f.dropout(f.relu(x), self.args.dropout, self.training)
        x = self.rnn(x)[0]
        x = f.dropout(f.relu(x), self.args.dropout, self.training)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
