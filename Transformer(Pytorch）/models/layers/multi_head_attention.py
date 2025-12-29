"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    # d_model = 512 n_head = 8
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()     # Attention机制 Q K V三个矩阵
        # Wq Wk Wv三个权重矩阵是d_model*d_model(512*512)的
        self.w_q = nn.Linear(d_model, d_model)          # Transformer训练的不是注意力权重分数，而是对于PE之后的256*512(max_len*d_model)Tensor转换为Q K V的权重矩阵Wq Wk Wv
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1.经过Embedding & Positional Encoding的Vector(256*512)经过Wq Wk Wv权重作用
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2.根据n_head = 8 进行Split操作,q k v都是(batch_size,max_len,d_model)->[batch, n_head, max_len, d_tensor]
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask) # 对每个head做Attention机制,得到加权之后的结果out 以及注意力权重attention

        # 4.concat and pass to linear layer
        # 把多头注意力得到的out拼接起来: [batch_size, head, length, d_tensor]->[batch_size, length, d_model]
        # 在经过一个Linear层得到out
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        # 根据多头注意力的heads个数进行拆分,论文中使用8 heads,则每一个Attention机制需要处理512/8=64维
        d_tensor = d_model // self.n_head
        # (batch_size,256(max_len),8(n_head),64),并且交换第1维度和第2维度信息 [batch, length, n_head, d_tensor] -> [batch, n_head, length, d_tensor]
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
