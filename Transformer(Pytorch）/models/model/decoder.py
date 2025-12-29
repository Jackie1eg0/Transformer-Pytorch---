"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

# Transformer的Decoder部分,需要和Encoder一样,Embedding 和 Positional Encoding
# 与Transformer中的Encoder不一样的是,Encoder是对源语言(英文)而Decoder是对目标语言(德文)
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,    # d_model = 512
                                        drop_prob=drop_prob, # drop_prob = 0.1
                                        max_len=max_len,    # max_len = 256
                                        vocab_size=dec_voc_size,    # dec_voc_size = 7859
                                        device=device)
        # 6层的DecoderLayer根据论文图示由三部分构成(三部分都使用到Res残差链接)
        # 1)Masked Multihead Attention 从[<sos>, A, B, <eos>] -> [<sos>, A, B ]作为Decoder的输入
        # 2)Multihead Attention 把Encoder的output 作为每一层的MutiheadAttention的Q K来源,把Decoder的input作为V的来源(通过乘以一个Wq Wk Wv)
        # 3)FeedForward前馈神经层
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])
        # 经过一个Linear层把6层的DecoderLayer的经过(batch_size, max_len,d_model)-> 变成(batch_size,max_len,dec_voc_size)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # 经过Embedding
        trg = self.emb(trg)
        # 经过6层的DecoderLayers作用,
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 最后的Decoder的结果会映射到dec_vac_size 7859上
        # (batch_size,max_len,d_model)-> (batch_size,max_len,dec_vac_size)
        output = self.linear(trg)
        return output