"""
实验 4: 分层记忆容量管理
优化版: 大batch + 多进程 + 大模型
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
    hidden_dim = 512  # 增大
    state_dim = 128   # 增大
    num_levels = 3    # 分层容量
    max_seq_len = 128
    batch_size = 256  # 大batch
    learning_rate = 1e-3
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=50000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.roll(self.data, shifts=-1, dims=1)
        self.labels[:, -1] = 0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LinearStateLayer(nn.Module):
    """稳定版状态层"""
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


class HierarchicalMemory(nn.Module):
    """分层记忆系统 - 容量管理"""
    def __init__(self, hidden_dim, state_dim, num_levels=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_levels = num_levels
        
        # 每层容量: 256, 1024, 4096 tokens
        self.capacities = [256, 1024, 4096]
        self.compression_ratios = [0.5, 0.25, 0.125]
        
        # 每层压缩器
        self.compressors = nn.ModuleList()
        self.decompressors = nn.ModuleList()
        self.state_layers = nn.ModuleList()
        
        for i in range(num_levels):
            compressed_dim = max(64, int(hidden_dim * self.compression_ratios[i]))
            self.compressors.append(nn.Sequential(
                nn.Linear(hidden_dim, compressed_dim),
                nn.ReLU(),
                nn.Linear(compressed_dim, compressed_dim)
            ))
            self.decompressors.append(nn.Sequential(
                nn.Linear(compressed_dim, hidden_dim),
                nn.ReLU()
            ))
            self.state_layers.append(LinearStateLayer(hidden_dim, state_dim, dropout))
        
        # 层级融合
        self.level_gates = nn.Linear(hidden_dim, num_levels)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        level_outputs = []
        for level in range(self.num_levels):
            # 压缩 -> 状态处理 -> 解压
            compressed = self.compressors[level](x)
            decompressed = self.decompressors[level](compressed)
            state_out, _ = self.state_layers[level](decompressed)
            level_outputs.append(state_out)
        
        # 动态层级权重
        gates = torch.softmax(self.level_gates(x.mean(dim=1, keepdim=True)), dim=-1)
        
        # 加权融合
        output = sum(gates[:, :, i:i+1] * level_outputs[i] for i in range(self.num_levels))
        output = self.norm(output + x)
        
        return output, gates.mean(dim=1)


class CapacityDHSM(nn.Module):
    """容量管理DHSM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        self.hierarchical_memory = HierarchicalMemory(
            config.hidden_dim, config.state_dim, 
            num_levels=config.num_levels, dropout=0.1
        )
        
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, level_weights = self.hierarchical_memory(x)
        return self.head(x), level_weights, None


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    total_weights = torch.zeros(3)
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, weights, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, weights, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()
        total_weights += weights.mean(dim=0).cpu()
    
    return total_loss / total_tokens, total_weights / len(dataloader)


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
    print("实验 4: 分层记忆容量管理 (优化版)")
    print("优化: 大batch(256) + 大模型(512dim) + 多进程")
    print("=" * 60)
    
    device = Config.device
    print(f"Device: {device}")
    print(f"Batch size: {Config.batch_size}")
    
    model = CapacityDHSM(Config()).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # 数据集 - 更大
    train_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=100000)
    val_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=10000)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, 
                            num_workers=2, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate * 0.5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_val_loss = float('inf')
    
    print("\n[Training]")
    for epoch in range(Config.num_epochs):
        train_loss, weights = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"Epoch {epoch+1}/{Config.num_epochs} | NaN detected!")
            break
        
        w_str = ' | '.join([f"L{i+1}:{w:.2f}" for i, w in enumerate(weights)])
        print(f"Epoch {epoch+1}/{Config.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {w_str}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/root/autodl-tmp/dhsm-research/experiments/best_model_exp4.pt')
    
    print("\n" + "=" * 60)
    print("实验 4 完成!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
