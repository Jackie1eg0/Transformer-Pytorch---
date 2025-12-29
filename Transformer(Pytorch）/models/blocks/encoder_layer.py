"""
@author : Hyunwoong
@when : 2019-10-24
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

# Encoder模块对应Transformer架构的左边
class EncoderLayer(nn.Module):
    # d_model = 512 ffn_hidden = 2048 
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # d_model = 512 n_head = 8 Multihead 具有8个heads, 可以得到[batch_size, length, d_model]
        # self.attention首先经过Wq Wk Wv得到Q K V矩阵(max_len,d_model)再经过Split送入Multihead_Attention(n_heads=8) [batch_size, 8, length, d_model/8]
        # 经过8个Attention机制的out再concat到[batch_size, length, d_model] 最后经过一个Linear层得到self.attention [batch_size,length,d_model]
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head) 
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1.由于要进行残差链接，因此需要先_x=x，再对输入Vector进行MultiheadAttention计算
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2.对应于论文示意图的Add & Norm 残差链接 & LayerNorm
        x = self.dropout1(x)
        x = self.norm1(x + _x)  # 直接相加送入LayerNorm
        
        # 3. FeedForward模块也需要进行残差链接 _x=x,再让x经过FeedForward
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
