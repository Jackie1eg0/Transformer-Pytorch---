"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math

from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Query(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity(按照论文公式 Q与K做内积得到注意力权重,之后把注意力权重乘以V矩阵)
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)(根据mask的需要选择是否做掩码)
        # src_mask 的作用。它的全称应该叫 Padding Mask 因为输入的Token长短不一,需要做Padding,如[ "I",   "love", "AI",   "<PAD>" ]
        # 如果不加限制，模型会算出 "I" 和 "<PAD>" 的相似度,这会导致 "I" 的注意力被分散给了一个毫无意义的占位符。Mask 矩阵告诉模型:在进入 Softmax 之前,把第 4 个位置的分数强行修改为 负无穷大 (-inf)。
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        # Padding 部分由于设置了-INF权重会趋向于0，因此注意力权重会很小。
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
