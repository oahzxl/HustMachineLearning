import torch
import torch.nn as nn
import torch.nn.functional as f


class Transformer(nn.Module):
    def __init__(self, args, vocab):
        super(Transformer, self).__init__()
        self.args = args
        self.emb = nn.Embedding(vocab.vocab.__len__(), args.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_model, nhead=2)
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc1 = nn.Conv1d(args.d_model, args.d_model >> 2, kernel_size=1, padding=0, stride=1)
        self.fc2 = nn.Conv1d((args.d_model >> 2) * args.max_words, 3, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.emb(x)
        x = x.transpose(0, 1)
        x = self.tf(x)
        x = f.dropout(x.transpose(0, 1), self.args.dropout, self.training)
        x = self.fc1(x.transpose(-1, -2))
        x = x.reshape(x.size(0), -1, 1)
        x = self.fc2(x).squeeze(-1)
        return x
