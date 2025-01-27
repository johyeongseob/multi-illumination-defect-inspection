import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_embed, max_len=7):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(torch.log(torch.tensor(10000.0)) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).cuda()


    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out

if __name__ == "__main__":
    pos_encoding = PositionalEncoding(d_embed=512)
    x = torch.zeros(8, 7, 512).cuda()
    output = pos_encoding(x)
    print(output)