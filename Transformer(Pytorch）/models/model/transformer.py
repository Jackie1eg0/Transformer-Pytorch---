"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder

# Seq2Seq Transformer架构模型, 在train.py中调用
# conf.py中设置了model的参数:d_model为512维 word2vec映射成512维的向量
# n_heads=8 multihead个数为8 n_layers=6 encoder和decoder堆叠个数为6
class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx  # src_pad_idx = 1
        self.trg_pad_idx = trg_pad_idx  # trg_pad_idx = 1
        self.trg_sos_idx = trg_sos_idx  # trg_sos_idx = 2
        self.device = device
        self.encoder = Encoder(d_model=d_model, # d_model = 512
                               n_head=n_head,   # n_head = 8 
                               max_len=max_len, # max_len = 256
                               ffn_hidden=ffn_hidden,# ffn_hidden = 2048
                               enc_voc_size=enc_voc_size,  # enc_voc_size = 5921
                               drop_prob=drop_prob, # drop_prob = 0.1
                               n_layers=n_layers,   # n_layers = 6
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,   # dec_voc_size = 7859
                               drop_prob=drop_prob,
                               n_layers=n_layers,           # n_layers = 6 和Encoder一样Decoder也是需要堆叠6层
                               device=device)

    def forward(self, src, trg):
        # src是Encoder的输入是原文,trg是Decoder的输入是译文
        # 例如src是[A, B, eos,<pad>,<pad>...]
        # 而译文trg是[sos C D E eos,<pad>,<pad>...]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # 经过Encoder与Decoder作用
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
    # 根据原文以及pad的token id去生成src_mask矩阵
    # 不看Padding Encoder最常见的输入是[A,B,eos,<pad>,<pad>...]
    # 变成[Batch, 1, 1, Src_Len] 配合Attention [batch_size,n_heads,max_len,d_model]进行Broadcasting
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
    # 根据译文 pad的token id 以及 下三角矩阵去生成trg_mask
    # Padding Mask 忽略<Pad>
    # Look-ahead Mask对应trg_sub_mask制造左下三角矩阵，非零元素都是1
    # 最后把trg_sub_mask和trg_pad_mask做相与操作,得到综合的mask矩阵
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask