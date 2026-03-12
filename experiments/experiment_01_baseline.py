"""
实验 1: 线性状态层基准测试
目标: 验证 SSM 组件能否学习简单序列模式
"""

import sys
sys.path.append('/root/autodl-tmp/dhsm-research/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import time
from dhsm_core import DHSM, DHSMConfig

# 配置
class Config:
    # 小规模实验
    vocab_size = 10000
    hidden_dim = 256
    num_layers = 4
    state_dim = 64
    num_heads = 4
    max_seq_len = 128
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleDataset(Dataset):
    """简单文本数据集 (模拟)"""
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.roll(self.data, shifts=-1, dims=1)
        self.labels[:, -1] = 0  # 移位后的填充
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    # tqdm disabled by request
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()
        
        
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        logits, _, _ = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()
    
    return total_loss / total_tokens


def benchmark_forward(model, batch_size, seq_len, device, iterations=100):
    """前向传播基准测试"""
    model.eval()
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # Warm up
    for _ in range(5):
        _, _, _ = model(x)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    for _ in range(iterations):
        _, _, _ = model(x)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start
    
    return elapsed / iterations


def benchmark_backward(model, batch_size, seq_len, device, iterations=10):
    """反向传播基准测试"""
    model.train()
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    criterion = nn.CrossEntropyLoss()
    
    # Warm up
    for _ in range(2):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        logits, _, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    
    for _ in range(iterations):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        logits, _, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
    
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start
    
    return elapsed / iterations


def main():
    print("=" * 60)
    print("实验 1: 线性状态层基准测试")
    print("=" * 60)
    
    device = Config.device
    print(f"Device: {device}")
    
    # 创建模型
    config = DHSMConfig(
        vocab_size=Config.vocab_size,
        hidden_dim=Config.hidden_dim,
        num_layers=Config.num_layers,
        state_dim=Config.state_dim,
        num_heads=Config.num_heads,
        max_seq_len=Config.max_seq_len,
        dynamic_depth=False
    )
    
    model = DHSM(config).to(device)
    params = model.count_parameters()
    print(f"Model parameters: {params['total_M']:.2f}M")
    
    # 数据集
    train_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=50000)
    val_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=5000)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, num_workers=0)
    
    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_val_loss = float('inf')
    
    print("\n[Training]")
    for epoch in range(Config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{Config.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/root/autodl-tmp/dhsm-research/experiments/best_model.pt')
    
    # 基准测试
    print("\n[Benchmark]")
    model.load_state_dict(torch.load('/root/autodl-tmp/dhsm-research/experiments/best_model.pt'))
    
    for bs in [8, 16, 32]:
        for sl in [64, 128]:
            fwd_time = benchmark_forward(model, bs, sl, device)
            print(f"Forward ({bs}x{sl}): {fwd_time*1000:.2f}ms")
    
    bwd_time = benchmark_backward(model, 16, 64, device)
    print(f"Backward (16x64): {bwd_time*1000:.2f}ms")
    
    print("\n" + "=" * 60)
    print("实验 1 完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
