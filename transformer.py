import torch
import torch.nn as nn
import math
import numpy as np
from make_data import *

d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8
dropout = 0.1   # dropout每次屏蔽10%

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer,self).__init__()
        self.enc_emb_input = Embedding(src_vocab_size)
        self.enc_pe_input = PositionalEncoding()
        self.enc_output = Encoder()
        self.dec_emb_input = Embedding(tgt_vocab_size)
        self.dec_pe_input = PositionalEncoding()
        self.dec_output = Decoder()
        self.output = Generator(d_model, tgt_vocab_size)

    def forward(self,enc_input, dec_input):
        enc_mask = get_attn_pad_mask(enc_input, enc_input)
        dec_mask = get_attn_subsequence_mask(dec_input)
        enc_emb_input = self.enc_emb_input(enc_input)
        enc_pe_input = self.enc_pe_input(enc_emb_input)
        enc_output = self.enc_output(enc_pe_input, enc_mask)
        dec_emb_input = self.dec_emb_input(dec_input)
        dec_pe_input = self.dec_pe_input(dec_emb_input)
        dec_output = self.dec_output(enc_output, dec_pe_input, dec_mask)
        output = self.output(dec_output)

        return output
    
class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self,x):
        return self.lut(x) #* math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.projection = nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()
        
    
    def forward(self, query, key, value, mask):
        scores = torch.matmul(query,key.transpose(-1, -2))/math.sqrt(d_k)
        scores.masked_fill_(mask, -1e9)
        scores = torch.matmul(scores,value)
        return torch.log_softmax(scores, dim=-1)

class Multi_Head_Attention(nn.Module):
    def __init__(self):
        super(Multi_Head_Attention, self).__init__()
        self.W_kq = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.SDPA = Scaled_Dot_Product_Attention()

    def forward(self, query, key, value, mask):
        Q = self.W_kq(query)
        K = self.W_kq(key)
        V = self.W_v(value)

        scores = self.SDPA(Q, K, V, mask)
        return scores
    
class Layer_Norm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(Layer_Norm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dropout=0.1):
        super(Position_wise_Feed_Forward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Encoder_layer(nn.Module):
    def __init__(self):
        super(Encoder_layer, self).__init__()
        self.MHA = Multi_Head_Attention()
        self.FF = Position_wise_Feed_Forward()

    def forward(self, enc_input, mask):
        x_1 = self.MHA(enc_input, enc_input, enc_input, mask)
        x_2 = nn.LayerNorm(d_model)(x_1 + enc_input)
        x_3 = self.FF(x_2)
        x_4 = nn.LayerNorm(d_model)(x_2 + x_3)
        return x_4
    
class Decoder_layer(nn.Module):
    def __init__(self):
        super(Decoder_layer, self).__init__()
        self.MHA = Multi_Head_Attention()
        self.FF = Position_wise_Feed_Forward()

    def forward(self, enc_output, dec_input, mask):
        x_1 = self.MHA(dec_input, dec_input, dec_input, mask)
        x_2 = nn.LayerNorm(d_model)(x_1 + dec_input)
        x_3 = self.MHA(enc_output, enc_output, x_2, mask)
        x_4 = nn.LayerNorm(d_model)(x_2 + x_3)
        x_5 = self.FF(x_4)
        x_6 = nn.LayerNorm(d_model)(x_4 + x_5)
        return x_6
    
# nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([Encoder_layer() for _ in range(n_layers)])

    def forward(self, enc_input, mask):
        for layer in self.layers:
            enc_outputs = layer(enc_input, mask)
        return enc_outputs
    
def get_attn_pad_mask(seq_q, seq_k):                                # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)                   # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)           # 扩展成多维度


def get_attn_subsequence_mask(seq):                                 # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)            # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # [batch_size, tgt_len, tgt_len]
    subsequence_mask = subsequence_mask.data.eq(0)
    return subsequence_mask

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([Decoder_layer() for _ in range(n_layers)])
    def forward(self, enc_output, dec_input, mask):
        for layer in self.layers:
            dec_outputs = layer(enc_output, dec_input, mask)
        return dec_outputs.view(-1, dec_outputs.size(-1))
