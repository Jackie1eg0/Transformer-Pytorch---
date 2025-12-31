"""
@author : YourName & Gemini
@desc   : Robust Inference script for Transformer
"""
import torch
import os
import re # 导入正则模块，用于更好的分词

# 导入所有配置 (d_model, n_layers, loader, etc.)
# 确保 data.py 与此文件在同一目录下
from data import * 
from models.model.transformer import Transformer

# ----------------------------------------------------------------------
# 1. 辅助函数：安全处理词表 (解决 'dict' has no attribute 'stoi')
# ----------------------------------------------------------------------
def get_vocab_mappings(loader):
    """
    自动判断 loader.source.vocab 是对象还是字典，
    并确保返回可用的 stoi (单词转数字) 和 itos (数字转单词)。
    """
    # --- Source (源语言) ---
    src_vocab = loader.source.vocab
    if isinstance(src_vocab, dict):
        src_stoi = src_vocab.get('stoi', src_vocab)
    else:
        src_stoi = src_vocab.stoi

    # --- Target (目标语言) ---
    trg_vocab = loader.target.vocab
    if isinstance(trg_vocab, dict):
        trg_stoi = trg_vocab.get('stoi', trg_vocab)
        # 如果字典里没有 itos，手动翻转生成 {id: word}
        if 'itos' in trg_vocab:
            trg_itos = trg_vocab['itos']
        else:
            trg_itos = {v: k for k, v in trg_stoi.items()}
    else:
        trg_stoi = trg_vocab.stoi
        trg_itos = trg_vocab.itos

    return src_stoi, trg_stoi, trg_itos

# ----------------------------------------------------------------------
# 2. 核心函数：句子预处理 (增加标点符号处理 & 调试打印)
# ----------------------------------------------------------------------
def preprocess_sentence(sentence, src_stoi, device):
    # --- A. 文本清洗 ---
    sentence = sentence.lower().strip()
    # 【改进点】: 在标点符号 [.!,?] 前后加空格，防止 "word." 被当成一个词
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    # 去除多余空格
    sentence = re.sub(r'[" "]+', " ", sentence)
    
    tokens = sentence.split()
    
    # --- B. 调试信息 (查看有多少词变成了 <unk>) ---
    unk_idx = src_stoi.get('<unk>', src_stoi.get('<pad>', 0))
    
    print(f"\n[Token Check]")
    print(f"Original: '{sentence}'")
    
    src_indexes = []
    unk_count = 0
    
    for token in tokens:
        if token in src_stoi:
            idx = src_stoi[token]
            print(f"  ✓ '{token}' -> ID {idx}")
        else:
            idx = unk_idx
            unk_count += 1
            print(f"  ? '{token}' -> <unk> (ID {idx}) [Not in Vocab]")
        src_indexes.append(idx)
            
    if unk_count > 0:
        print(f"[Warning] Found {unk_count} unknown words. Translation may be inaccurate.")

    # --- C. 添加 <sos>, <eos> 并转 Tensor ---
    # 兼容不同的键名 (有些词表用 '<sos>', 有些用 'sos')
    sos_token = '<sos>' if '<sos>' in src_stoi else 'sos'
    eos_token = '<eos>' if '<eos>' in src_stoi else 'eos'
    
    src_indexes = [src_stoi[sos_token]] + src_indexes + [src_stoi[eos_token]]
    
    # [1, seq_len]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    return src_tensor

# ----------------------------------------------------------------------
# 3. 核心函数：模型加载
# ----------------------------------------------------------------------
def load_trained_model(model_path, device):
    print(f"Loading model architecture (d_model={d_model}, layers={n_layers})...")
    model = Transformer(src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx,
                        trg_sos_idx=trg_sos_idx,
                        d_model=d_model,
                        enc_voc_size=enc_voc_size,
                        dec_voc_size=dec_voc_size,
                        max_len=max_len,
                        ffn_hidden=ffn_hidden,
                        n_head=n_heads,
                        n_layers=n_layers,
                        drop_prob=drop_prob,
                        device=device).to(device)

    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        print(f"File not found: {model_path}")
        return None

# ----------------------------------------------------------------------
# 4. 核心函数：翻译循环 (Greedy Decode)
# ----------------------------------------------------------------------
def translate(model, src_tensor, trg_stoi, trg_itos, device):
    sos_token = '<sos>' if '<sos>' in trg_stoi else 'sos'
    eos_token = '<eos>' if '<eos>' in trg_stoi else 'eos'
    
    sos_idx = trg_stoi[sos_token]
    eos_idx = trg_stoi[eos_token]
    
    # 初始输入: [<sos>]
    trg_indexes = [sos_idx]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
            # 取最后一个预测
            pred_token_idx = output[:, -1, :].argmax(dim=-1).item()
            
        if pred_token_idx == eos_idx:
            break
            
        trg_indexes.append(pred_token_idx)
    
    # 转换为单词 (跳过开头 <sos>)
    trg_tokens = [trg_itos.get(idx, '<unk>') for idx in trg_indexes[1:]]
    return " ".join(trg_tokens)

# ----------------------------------------------------------------------
# 主程序入口
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 自动选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # === 请在这里修改你的模型文件名 ===
    model_path = 'weight.pt' # 或者 'saved/model-xxxx.pt'
    
    # 1. 准备词表
    try:
        src_stoi, trg_stoi, trg_itos = get_vocab_mappings(loader)
    except NameError:
        print("Error: 'loader' not found. Please check data.py.")
        exit()

    # 2. 加载模型
    model = load_trained_model(model_path, device)
    
    if model is not None:
        print("\n" + "="*40)
        print("   Transformer Translator Ready")
        print("   (Type 'q' to quit)")
        print("="*40)
        
        while True:
            try:
                text = input("\nEnter English sentence: ")
                if text.lower() in ['q', 'quit', 'exit']:
                    break
                
                if not text.strip():
                    continue

                # 预处理
                src_tensor = preprocess_sentence(text, src_stoi, device)
                
                # 翻译
                result = translate(model, src_tensor, trg_stoi, trg_itos, device)
                
                print(f"Translation: {result}")
                
            except Exception as e:
                print(f"Error during translation: {e}")