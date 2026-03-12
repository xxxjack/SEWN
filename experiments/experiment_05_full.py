"""
实验5: 完整DHSM - 综合优化版
结合实验1(单层稳定)+实验3(动态计算)+大batch优化
"""

import sys
sys.path.append('/root/autodl-tmp/dhsm-research/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Config:
    vocab_size = 10000
    hidden_dim = 512
    state_dim = 96
    max_seq_len = 128
    batch_size = 256
    learning_rate = 1e-3
    num_epochs = 15  # 多跑几个epoch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=100000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.roll(self.data, shifts=-1, dims=1)
        self.labels[:, -1] = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LinearStateLayer(nn.Module):
    """稳定版状态层 - 基于实验1"""
    def __init__(self, hidden_dim, state_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.A_log = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.B = nn.Linear(hidden_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, hidden_dim, bias=False)
        self.D = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        
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
            self.A_log.data = torch.log(torch.ones(self.state_dim) * 0.95 + 0.02 * torch.randn(self.state_dim))
    
    def forward(self, x, state=None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        A = torch.exp(self.A_log).clamp(0.5, 0.99)
        
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=device)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            gate = self.gate(x_t).clamp(0, 1)
            Bx = self.B(x_t)
            state = A * state + gate * Bx
            state = torch.clamp(state, -10, 10)
            y_t = self.C(state) + self.D * x_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        output = self.norm(output + x)
        output = self.dropout(self.out_proj(output))
        
        return output, state


class FullDHSM(nn.Module):
    """完整DHSM - 单层+残差+大batch优化"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding with positional
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.hidden_dim) * 0.02)
        
        # 主干 - 单层稳定状态层 (实验1)
        self.state_layer = LinearStateLayer(config.hidden_dim, config.state_dim, dropout=0.15)
        
        # 残差连接
        self.res_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
    
    def forward(self, x):
        # Embedding + position
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        
        # 残差备份
        residual = self.res_proj(x)
        
        # 状态层处理
        output, state = self.state_layer(x)
        
        # 残差连接
        output = output + residual
        output = self.final_norm(output)
        
        return self.head(output), state, None


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits, _, _ = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()
    
    return total_loss / total_tokens


def main():
    print("=" * 60)
    print("实验 5: 完整DHSM - 综合优化版")
    print("结合: 单层稳定(实验1) + 大batch优化")
    print("=" * 60)
    
    device = Config.device
    print(f"Device: {device}")
    print(f"Batch size: {Config.batch_size}")
    print(f"Hidden dim: {Config.hidden_dim}")
    
    model = FullDHSM(Config()).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # 数据集
    train_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=150000)
    val_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=15000)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, 
                            num_workers=2, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_val_loss = float('inf')
    patience = 5
    no_improve = 0
    
    print("\n[Training]")
    for epoch in range(Config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"Epoch {epoch+1}/{Config.num_epochs} | NaN detected!")
            break
        
        print(f"Epoch {epoch+1}/{Config.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/root/autodl-tmp/dhsm-research/experiments/best_model_exp5.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 60)
    print("实验 5 完成!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
