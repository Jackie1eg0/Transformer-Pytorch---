"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

# Transformer架构中的Encoder层
class Encoder(nn.Module):
    
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # Embedding部分:含有token的embedding & Positional Encoding
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size, # vocab_size = 5921
                                        drop_prob=drop_prob,
                                        device=device)
        # 使用ModuleList创建n_layers(6)层的EncoderLayer(EncoderLayer分为MutiheadAttention 以及FeedForward两部分都需要Res残差链接)
        # EncoderLayer堆叠6个即可每个EncoderLayer结构都是一样的
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    # Transformer Encoder部分的前向传播:分为Embedding PositionEncoding(self.emb部分) & 6层的EncoderLayer(self.layers部分)
    def forward(self, x, src_mask):
        x = self.emb(x)
        # self.layers类似于一个拥有6个EncoderLayer实例索引列表,Encoder前向传播经过6个Encoder Layers
        for layer in self.layers:
            x = layer(x, src_mask)
        # 最终得到的输出[batch_size,max_len,d_model]再送入Decoder部分
        return x