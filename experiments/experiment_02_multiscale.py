"""
实验 2: 多尺度状态空间
目标: 验证多层级状态表示的效果
"""

import sys
sys.path.append('/root/autodl-tmp/dhsm-research/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dhsm_core import DHSM, DHSMConfig, HierarchicalMemory

class Config:
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
    def __init__(self, vocab_size, seq_len, num_samples=10000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.roll(self.data, shifts=-1, dims=1)
        self.labels[:, -1] = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MultiScaleDHSM(nn.Module):
    """多尺度 DHSM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # 多尺度状态层
        self.state_layers = nn.ModuleList([
            LinearStateLayer(config.hidden_dim, config.state_dim * (2 ** i), config.dropout)
            for i in range(3)  # 3 个尺度
        ])
        
        # 状态融合
        self.fusion = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # 输出层
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
    
    def forward(self, x):
        emb = self.embedding(x)
        
        # 多尺度处理
        states = []
        for layer in self.state_layers:
            out, state = layer(emb)
            states.append(out)
        
        # 融合多尺度
        combined = torch.cat(states, dim=-1)
        fused = self.fusion(combined)
        fused = self.norm(fused + emb)
        
        return self.head(fused), None, None


class LinearStateLayer(nn.Module):
    """简化版状态层"""
    def __init__(self, hidden_dim, state_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.A_log = nn.Parameter(torch.randn(state_dim))
        self.B = nn.Linear(hidden_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, hidden_dim, bias=False)
        self.D = nn.Parameter(torch.randn(hidden_dim))
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init()
    
    def _init(self):
        with torch.no_grad():
            self.A_log.data = torch.log(torch.exp(-0.5 * torch.ones(self.state_dim)) + 0.1 * torch.randn(self.state_dim))
    
    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        A = torch.exp(self.A_log)
        
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=device)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            gate = self.gate(x_t)
            Bx = self.B(x_t)
            state = A * state + gate * Bx
            y_t = self.C(state) + self.D * x_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        output = self.norm(output + x)
        output = self.dropout(self.out_proj(output))
        
        return output, state


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    
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


def main():
    print("=" * 60)
    print("实验 2: 多尺度状态空间")
    print("=" * 60)
    
    device = Config.device
    print(f"Device: {device}")
    
    # 创建模型
    class ExpConfig:
        vocab_size = Config.vocab_size
        hidden_dim = Config.hidden_dim
        state_dim = Config.state_dim
        dropout = 0.1
    
    model = MultiScaleDHSM(ExpConfig()).to(device)
    
    # 估算参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
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
            torch.save(model.state_dict(), '/root/autodl-tmp/dhsm-research/experiments/best_model_exp2.pt')
    
    print("\n" + "=" * 60)
    print("实验 2 完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
