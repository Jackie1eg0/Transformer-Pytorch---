"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time
import os
from torch import nn, optim
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time

# 基于 PyTorch 实现的 Transformer 模型训练与评估脚本。
# 它涵盖了模型初始化、优化器配置、学习率调度、训练循环以及使用 BLEU 分数进行性能评估。

# 计算模型中可以训练的参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 初始化模型权重：使用 Kaiming Uniform (He初始化) 方法，有助于深层网络的收敛
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


# 对Transformer模型进行实例化
model = Transformer(src_pad_idx=src_pad_idx,    # src_pad_idx & trg_pad_idx (填充符号索引),添加pad用于把句子填充到一致的长度,短句子用pad
                    trg_pad_idx=trg_pad_idx,    # 模型根据trg_pad_idx判断句子中哪些是凑数的,算Attention无视这些位置

                    trg_sos_idx=trg_sos_idx,    # (Start of Sentence)表示开始,Decoder 的第一个输入。
                                                # 在推理（翻译）时,没有任何前文。你必须先喂给Decoder一个<sos>，它才会吐出翻译的第一个词（比如“我”） 
                    d_model=d_model,            # 512维 vector是512维

                    enc_voc_size=enc_voc_size,  # 统计词典英语 德语的词表大小统计,此处有5912个
                                                # (Encoder Vocabulary Size)源语言(英语)词表大小，有多少个单词
                                                #  用于定义Embedding矩阵大小,enc_voc_size是10,000,d_model是512，那么 Encoder 入口处就会生成一个 [10000, 512] 的巨大矩阵，专门用来把Token ID -> Vector。
                    dec_voc_size=dec_voc_size,  # (Decoder Vocabulary Size)目标语言（德文）词表大小，此处有7859个
                    
                    max_len=max_len,            # 最大序列长度256 token长度限制,如empowers这个vocalbulary可以被分为2个token em powers
                                                #【Token 到底是什么？—— 揭秘大模型背后的“文字压缩术”】https://www.bilibili.com/video/BV1S5miBvEsu?vd_source=58cd116d4f3727402e47dc3abb530e6d
                    ffn_hidden=ffn_hidden,      # ffn_hidden = 2048 (前馈层隐藏维度)
                   
                    n_head=n_heads,             # Multihead 多头注意力机制 8 heads
                    n_layers=n_layers,          # Encoder Decoder堆叠的层数 6层堆叠
                    drop_prob=drop_prob,
                    device=device).to(device)

# 调用上面的count_parameters输出模型可训练的参数
# 应用初始化权重
# 定义Adam优化器
# 定义学习率调度器：当验证集损失不再下降时，降低学习率
# 定义损失函数：交叉熵损失，忽略用于填充的 token (pad_idx)
print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 factor=factor,
                                                 patience=patience)

# 完全忽略Padding的交叉熵损失,在Transformer当中,无论是翻译任务还是文本生成任务,本质都是一个多分类问题
# 输出是词表中每一个词成为下一个词的概率,若dec_val_size=7859那么本质是在做7859选1的多分类任务
# 模型预测的下一个词的输出是一个长度为val_size的概率向量,而真实标签是一个One-Hot[0,0,1,0,...],在实际中只有真实词位置k为1,其他位置的yi为0,,只需看真实词对于的概率就行,让其最大化
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)


def train(model, iterator, optimizer, criterion, clip):

# 参数model为Transformer对象(负责接受源语言索引序列 src 和目标语言索引序列 trg)并输出预测的概率分布。
# iterator 训练数据的迭代器（通常是 PyTorch 的 DataLoader)
# criterion 损失函数（代码中使用的是 CrossEntropyLoss）

    model.train()   # 训练模式,启用 Dropout 层（随机丢弃神经元）和 BatchNorm/LayerNorm 层的参数更新
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, trg = batch    # src: 源语言句子张量,trg: 目标语言句子张量
                            # [batch_size, src_len]与[batch_size, trg_len]
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  
        # 前向传播 假设target为[<sos>, A, B, <eos>] 模型输入(右下角outputs Shift Right) 为[<sos>, A, B]
        # 模型预测输出(右上角 Outputs Probabilities)，#output的形状是[512,trg_len,7859](batch_size在conf.py设置为512,trg_len是当前batch句子最长token大小,7859是dec_val_size) 
        
        print("经过Transformer的Output形状为:")
        print(output.size()) 
        #中间那个维度代表“当前这个 Batch 中最长句子的token长度,DataLoader采用动态填充机制,防止强制填充到max_len=256,计算冗余
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        print("经过reshape之后的Output形状为:")
        print(output_reshape.size())
        # 原先output形状:[batch_size, trg_len - 1, dec_val_size] ->[(batch_size * (trg_len - 1)), dec_val_size]
        # batch_size一次处理多少个句子(512),trg_len-1在这个batch中最长的句子具有Token个数(去掉开头<sos>) dec_val_size德语词表长度,对每个单词作为下一个单词出现可能性概率打分
        print("trg的形状为")
        print(trg.size())
        trg = trg[:, 1:].contiguous().view(-1)
        # trg应该是token ID序列(batch_size, trg_len)      [<sos> TokenID1 TokenID2 ... <pad>](<sos>与<pad>也会有对应的TokenID)
        print("经过处理后的trg形状为")
        print(trg.size())
        # 处理后的trg去除<sos>并且变形为(batch*trg_len)一维的tensor
        
        loss = criterion(output_reshape, trg)
        # 做交叉熵反向传播,将outputs预测的输出与真实的标签trg, (batch_size,trg_len,dec_val_size)与(batch_size*trg_len , dec_val_size)做交叉熵
        # 相当于从每一个Token视角他有TokenID 以及经过Transformer输出的dec_val_size预测可以做交叉熵
        # 做交叉熵时候会先做 Softmax 把分数变成概率。
        # 特别提醒：那个烦人的 Padding,后面有很多是填充符<pad>(ID=1)计算Loss时也会计算？
        # ----->>初始化时的关键参数 criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx) # src_pad_idx 通常是 1
        # ----->>虽然形式上是(18944, 7859)和(18944)做交叉熵，但有ignore_index,实际上只有那些非Padding的有效单词贡献了梯度和Loss,<pad>被完美地忽略了
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss / len(iterator)

# 评估模型性能性能的评估有两个维度：Loss（损失值） 和 BLEU Score（翻译质量指标）
def evaluate(model, iterator, criterion):
    model.eval()    
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch
            # target为[<sos>, A, B, <eos>] model的输入[<sos>, A, B]
            # model期望的预测输出为[A, B, <eos>] 真实标签trg_flat为[A, B, <eos>]
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg_flat)
            epoch_loss += loss.item()

            # BLEU 分数计算
            total_bleu = []
            for j in range(trg.shape[0]): # 遍历当前 Batch 中的每一句话
                try:
                    # 这是一个辅助函数（在 util.bleu 中定义）。它负责TokenID如 [4, 25, 11, 3] 转换回文本字符串（如 "i love you <eos>"）。
                    # 通常它还会处理掉 <pad> 等特殊符号。
                    trg_words = idx_to_word(trg[j], loader.target)     
                    # output为(batch_size,trg_len,dec_val_size),在第j个句子的每一个位置上找到概率最大的TokenID
                    output_words = output[j].max(dim=1)[1]              
                    output_words = idx_to_word(output_words, loader.target)
                    # 根据预测的TokenID得到文本串与真实的文本串做BLEU损失：
                    # BLEU (Bilingual Evaluation Understudy) 分数的核心逻辑就是比较“预测出来的文本（Hypothesis）”和“真实标签文本（Reference）”的相似度。
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except Exception as e:
                    print(f"BLEU failure on sample No.{j} : {e}")

            if total_bleu:
                avg_bleu = sum(total_bleu) / len(total_bleu)
                batch_bleu.append(avg_bleu)

    batch_bleu_score = sum(batch_bleu) / len(batch_bleu) if batch_bleu else 0
    return epoch_loss / len(iterator), batch_bleu_score


def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []  
    for step in range(total_epoch):
        start_time = time.time()  
        train_loss = train(model, train_iter, optimizer, criterion, clip)  
        valid_loss, bleu = evaluate(model, valid_iter, criterion)  
        end_time = time.time()  

        if step > warmup: 
            scheduler.step(valid_loss)  
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current Learning Rate: {current_lr}')

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)  

        if valid_loss < best_loss and step % 50 == 0:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        # 检查 result 文件夹是否存在，如果不存在则创建
        if not os.path.exists('result'):
            os.makedirs('result')
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()

        f = open('result/bleu.txt', 'w')
        f.write(str(bleus))
        f.close()

        f = open('result/test_loss.txt', 'w')
        f.write(str(test_losses))
        f.close()

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)