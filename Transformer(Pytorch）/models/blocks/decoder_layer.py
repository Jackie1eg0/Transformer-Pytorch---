"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

# PS: Decoder的output经过Linear Softmax得到的output Probabilities预测结果应该是 [A, B, eos]
class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):     # d_model = 512 ffn_hidden = 2048 n_head = 8
        
        # 1)Decoder模块的第1个Masked Multihead Attention 
        # 其 Q K V都来自于目标语言[A, B ,eos](原先目标语言经过Shift Right 从[<sos>, A, B, <eos>]->[sos, A, B ]),
        # 并且配合Mask 在处理第0个位置的时候只能看见<sos> 处理第1个位置时候只能看见 <sos> A  
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        
        # 2)Decoder模块的第2个Multihead-Attention 
        # ******(注意目标语言dec用于生成提供Q,Encoder的outputs生成KV)
        # Q的来源是目标语言的[A, B, eos] 其 K V 的来源是经过6个EncoderLayer输出的output[batch_size,max_len,d_model] 
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        # 3)FeedForward层
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    # Transformer DecoderLayer的前向传播
    def forward(self, dec, enc, trg_mask, src_mask):
        # trg_mask: Decoder下三角掩码（防止看未来）,src_mask: Padding 掩码（防止关注源语言的填充符<pad>）。
        
        # 1)Masked Multihead Attention Q K V来源于目标语言[A,B,eos]
        #   trg_mask为左下三角矩阵,防止在生成当前位置时候偷看后面的文本
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        #   Masked Multihead Attention 残差连接与LayerNorm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 2)Multihead Attention Q来自于dec,  K V都来自于Encoder的outputs
        if enc is not None:
            # 把Decoder的目标语言作为Q, 6层的EncoderLayer输出outputs作为K V来源
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            # 使用src_mask为了忽略<pad>的注意力的计算,完全是为了消除 <pad> 的注意力影响。
            # 经过6层EncoderLayer的输入和输出的形状完全一样，位置严格对应。
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 3)FeedForward 前馈神经网络部分
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
