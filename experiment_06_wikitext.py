"""
实验6: DHSM基于真实数据训练 (WikiText-2)
海蓝 🌊 - 2026-03-12

基于实验1的成功经验 + 真实WikiText-2数据 + 更大模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import numpy as np
import math
import time
from pathlib import Path

# ==================== 配置 ====================
CONFIG = {
    'device': 'cuda',
    'max_seq_length': 512,
    'vocab_size': 10000,
    'hidden_dim': 768,  # 比实验1更大
    'state_dim': 128,   # 更大的状态维度
    'num_layers': 6,    # 多层堆叠
    'dropout': 0.1,
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'data_path': '/root/autodl-tmp/dhsm-research/data/wikitext2',
    'save_path': '/root/autodl-tmp/dhsm-research/experiments/',
}

# ==================== SSM状态层 (实验1验证) ====================
class StateSpaceLayer(nn.Module):
    """线性状态空间层 - 实验1最佳结构"""
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        
        # SSM参数
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, input_dim, bias=False)
        self.D = nn.Linear(input_dim, input_dim, bias=False)
        
        # 门控
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(CONFIG['dropout'])
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        residual = x
        batch_size, seq_len, _ = x.shape
        
        # 初始化状态
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            
            # 状态更新: s_t = A * s_{t-1} + B * x_t
            state = torch.matmul(state, self.A.T) + self.B(x_t)
            state = torch.clamp(state, -10, 10)  # 防止爆炸
            
            # 输出: y_t = C * s_t + D * x_t
            y_t = self.C(state) + self.D(x_t)
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, input_dim)
        
        # 门控融合
        gate = self.gate(x)  # (batch, seq_len, 2)
        output = gate[:, :, 0:1] * output + gate[:, :, 1:2] * residual
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output

# ==================== 完整模型 ====================
class DHSMLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, state_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(CONFIG['max_seq_length'], hidden_dim)
        
        # 多层SSM
        self.layers = nn.ModuleList([
            StateSpaceLayer(hidden_dim, state_dim)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, 0, 0.02)
        nn.init.normal_(self.pos_embedding.weight, 0, 0.02)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Embedding + Positional
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # SSM层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits}

# ==================== 数据集 ====================
class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.examples = []
        for text in texts:
            if len(text.strip()) < 10:  # 跳过太短的文本
                continue
            tokens = tokenizer.encode(text).ids
            if len(tokens) < 2:
                continue
            # 截断或填充
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {'input_ids': input_ids, 'labels': labels}

# ==================== 训练函数 ====================
def train():
    print("=" * 60)
    print("实验6: DHSM WikiText-2 真实数据训练")
    print("=" * 60)
    
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    
    # 加载数据
    print("\n[Loading Dataset]")
    dataset = load_from_disk(CONFIG['data_path'])
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Valid: {len(dataset['validation'])} samples")
    
    # 训练tokenizer
    print("\n[Training Tokenizer]")
    tokenizer_path = Path(CONFIG['save_path']) / "tokenizer_wikitext.json"
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print("Loaded existing tokenizer")
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG['vocab_size'],
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
        )
        
        def batch_iterator():
            for i in range(0, len(dataset['train']), 1000):
                yield dataset['train'][i:i+1000]['text']
        
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print("Tokenizer trained and saved")
    
    # 准备数据
    print("\n[Preparing Data]")
    train_texts = [item['text'] for item in dataset['train']]
    valid_texts = [item['text'] for item in dataset['validation']]
    
    train_dataset = WikiTextDataset(train_texts, tokenizer, CONFIG['max_seq_length'])
    valid_dataset = WikiTextDataset(valid_texts, tokenizer, CONFIG['max_seq_length'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=CONFIG['batch_size'],
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    
    # 创建模型
    print("\n[Creating Model]")
    model = DHSMLanguageModel(
        CONFIG['vocab_size'],
        CONFIG['hidden_dim'],
        CONFIG['state_dim'],
        CONFIG['num_layers']
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs']
    )
    
    # 训练循环
    print("\n[Training]")
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels)
            loss = outputs['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_train_loss = np.mean(train_losses)
        
        # Validate
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, labels)
                val_losses.append(outputs['loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Perplexity: {math.exp(avg_val_loss):.2f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_file = Path(CONFIG['save_path']) / "best_model_exp6.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'config': CONFIG,
            }, save_file)
            print(f"  ✓ Saved best model (Val: {best_val_loss:.4f})")
        
        scheduler.step()
    
    print("\n" + "=" * 60)
    print("实验6 完成!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Perplexity: {math.exp(best_val_loss):.2f}")
    print("=" * 60)

if __name__ == '__main__':
    train()
