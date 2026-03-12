#!/usr/bin/env python3
"""
实验8.5 - 学习率优化版
使用与实验8相同的tokenizer，添加Cosine退火 + Warmup
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast
import math
import time

# ============ 多世界SSM架构 ============

class WorldSSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))
        self.register_buffer('h0', torch.zeros(1, d_state))
        
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        A_discrete = torch.matrix_exp(self.A * 0.1)
        h = self.h0.expand(batch, -1, -1)
        outputs = []
        for t in range(seq_len):
            h = torch.bmm(h, A_discrete.unsqueeze(0).expand(batch, -1, -1))
            h = h + torch.bmm(x[:, t:t+1], self.B.unsqueeze(0).expand(batch, -1, -1))
            y = torch.bmm(h, self.C.unsqueeze(0).expand(batch, -1, -1)).squeeze(1)
            outputs.append(y + x[:, t] * self.D)
        return torch.stack(outputs, dim=1)


class MultiWorldSSM(nn.Module):
    def __init__(self, vocab_size, d_model=256, world_configs=[64, 128, 256, 512]):
        super().__init__()
        self.d_model = d_model
        self.n_worlds = len(world_configs)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.worlds = nn.ModuleList([WorldSSM(d_model, ds) for ds in world_configs])
        self.router = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, self.n_worlds), nn.Softmax(dim=-1))
        self.output = nn.Linear(d_model, vocab_size)
        self.register_buffer('world_usage', torch.zeros(self.n_worlds))
        
    def forward(self, x):
        batch, seq_len = x.shape
        x_emb = self.embedding(x)
        world_weights = self.router(x_emb.mean(dim=1))
        self.world_usage += world_weights.sum(dim=0)
        
        output = torch.zeros(batch, seq_len, self.d_model, device=x.device)
        for i, world in enumerate(self.worlds):
            output = output + world_weights[:, i].unsqueeze(1).unsqueeze(2) * world(x_emb)
        
        return self.output(output)


class TextDataset(Dataset):
    def __init__(self, data, seq_len=128):
        self.data = data
        self.seq_len = seq_len
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long),
            torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        )


class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


def train():
    print('=' * 60)
    print('实验8.5 - 学习率优化 (Cosine退火 + Warmup)')
    print('=' * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizer (与实验8相同)
    tokenizer_path = '/root/autodl-tmp/dhsm-research/experiments/tokenizer_wikitext.json'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f'词表大小: {vocab_size}')
    
    # 加载数据
    print('加载WikiText-103...')
    dataset = load_from_disk('/root/autodl-tmp/dhsm-research/data/wikitext103')
    
    # Tokenize
    def tokenize_batch(texts, max_total=2000000):
        all_tokens = []
        for text in texts:
            if len(all_tokens) >= max_total:
                break
            try:
                tokens = tokenizer.encode(text)[:512]
                if len(tokens) > 10:
                    all_tokens.extend(tokens)
            except:
                pass
        return all_tokens
    
    print('Tokenizing训练集...')
    train_tokens = tokenize_batch(dataset['train']['text'], max_total=2000000)
    print(f'训练tokens: {len(train_tokens)}')
    
    print('Tokenizing验证集...')
    val_tokens = tokenize_batch(dataset['validation']['text'], max_total=200000)
    print(f'验证tokens: {len(val_tokens)}')
    
    if len(train_tokens) < 1000 or len(val_tokens) < 100:
        print('数据不足，退出')
        return
    
    train_dataset = TextDataset(train_tokens, seq_len=128)
    val_dataset = TextDataset(val_tokens, seq_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4, pin_memory=True)
    
    # 模型
    model = MultiWorldSSM(vocab_size, d_model=256, world_configs=[64, 128, 256, 512]).to(device)
    
    # 加载实验8权重
    checkpoint_path = '/root/autodl-tmp/dhsm-research/experiments/best_model_exp8.pt'
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"✓ 加载实验8权重成功 (Epoch {ckpt.get('epoch', '?')}, PPL {ckpt.get('best_ppl', '?')})")
    except Exception as e:
        print(f'警告: 加载权重失败 - {e}')
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    total_steps = len(train_loader) * 5
    warmup_steps = len(train_loader) // 2
    scheduler = CosineAnnealingWarmup(optimizer, warmup_steps, total_steps, 1e-6)
    criterion = nn.CrossEntropyLoss()
    
    best_ppl = float('inf')
    start_time = time.time()
    log_file = '/root/autodl-tmp/dhsm-research/experiments/exp08b_output.log'
    
    # 清空日志
    with open(log_file, 'w') as f:
        f.write('实验8.5 - 学习率优化\n')
    
    for epoch in range(1, 6):
        model.train()
        train_loss = 0
        epoch_start = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            lr = scheduler.step()
            
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f'[Epoch {epoch}/5] {progress:.1f}% | Loss: {loss.item():.4f} | LR: {lr:.2e}', end='\r')
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x).view(-1, vocab_size), y.view(-1)).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        ppl = math.exp(min(val_loss, 10))
        
        epoch_time = (time.time() - epoch_start) / 60
        print(f'\nEpoch {epoch} 完成! Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | 耗时: {epoch_time:.1f}分钟')
        
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_ppl': ppl},
                      '/root/autodl-tmp/dhsm-research/experiments/best_model_exp8b.pt')
            print(f'  ✓ 保存最佳模型 (PPL: {ppl:.2f})')
        
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, PPL={ppl:.2f}\n')
    
    total_time = (time.time() - start_time) / 60
    print('=' * 60)
    print(f'实验8.5 完成! 最佳PPL: {best_ppl:.2f} | 总耗时: {total_time:.1f}分钟')
    print('=' * 60)
    
    with open(log_file, 'a') as f:
        f.write(f'\n最终: 最佳PPL={best_ppl:.2f}, 总耗时={total_time:.1f}分钟\n')


if __name__ == '__main__':
    train()
