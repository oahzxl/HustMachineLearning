import torch.nn as nn


class FC(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(vocab.vocab.__len__(), args.d_model)
        self.fc1 = nn.Linear(args.d_model, args.d_model >> 4)
        self.fc2 = nn.Linear((args.d_model >> 4) * args.max_words, 3)

    def forward(self, x):
        x = self.emb(x)
        x = self.fc1(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc2(x)
        return x
