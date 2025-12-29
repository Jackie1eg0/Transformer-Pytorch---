"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()
        
        # Positional Encoding位置编码 根据max_len = 256,d_model = 512 ,生成一个256*512矩阵与Embedding的结果相加
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  
        # 生成 0 到 max_len-1 的整数序列 并且 [max_len]->[max_len,1]
        # 1D => 2D unsqueeze to represent word's position
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
    
        # 生成 0, 2, 4, ... 510 的偶数序列
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
   
        # 对于每一个token都有 512维度的向量，其中对于偶数位置(0, 2, 4...)填sin,奇数位置 (1, 3, 5...)填cos
        # self.encoding是max_len*d_model 2Dtensor
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
