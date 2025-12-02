import torch
from torch import nn
import config
import math

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float)
        div_term = torch.pow(10000.0, _2i / self.d_model)
        sins = torch.sin(pos / div_term)
        coss = torch.cos(pos / div_term)

        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)
        pe[:, 0::2] = sins
        pe[:, 1::2] = coss
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x的shape batch_size, sql_len, d_model
        seq_len = x.shape[1]
        pe_part = self.pe[:seq_len]
        return x + pe_part

class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.scr_embedding = nn.Embedding(num_embeddings=zh_vocab_size, 
                                          embedding_dim=config.DIM_MODEL,
                                          padding_idx=zh_padding_index)
        self.tgt_embedding = nn.Embedding(num_embeddings=en_vocab_size,
                                          embedding_dim=config.DIM_MODEL,
                                          padding_idx=en_padding_index)

        self.position_encoding = PositionEncoding(d_model=config.DIM_MODEL)
        
        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                          nhead=config.NUM_HEADS,
                                          num_encoder_layers=config.NUM_ENCONDER_LAYERS,
                                          num_decoder_layers=config.NUM_DECODER_LAYERS,
                                          batch_first=True)
        self.linear = nn.Linear(in_features=config.DIM_MODEL,
                                out_features=en_vocab_size)

    def encode(self, src, src_pad_mask):
        # src.shape batch_size, seq_len
        src_embed = self.scr_embedding(src)
        # src_embed.shape batch_size, seq_len, d_model
        src_embed = self.position_encoding(src_embed)
        # 送入transformer
        memory = self.transformer.encoder(src=src_embed, src_key_padding_mask=src_pad_mask)
        # memory.shape batch_size, seq_len, d_model
        return memory
    
    def decode(self, tgt, memory, memory_pad_mask, tgt_pad_mask, tgt_mask):
        tgt_embed =self.position_encoding(self.tgt_embedding(tgt))
        output = self.transformer.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask, 
                                 memory_mask=memory_pad_mask, tgt_key_padding_mask=tgt_pad_mask, 
                                 memory_key_padding_mask=memory_pad_mask)
        # output.shape batch_size, seq_len, d_model
        output = self.linear(output)
        # output.shape batch_size, seq_len, en_vocab_size
        return output
    
    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_mask, memory_pad_mask):
        memory = self.encode(src, src_pad_mask)
        output = self.decode(tgt, memory, memory_pad_mask, tgt_pad_mask, tgt_mask)
        return output

if __name__ == '__main__':
    model = TranslationModel(zh_vocab_size=100, en_vocab_size=100, zh_padding_index=0, 
                            en_padding_index=0)
    print(model)


