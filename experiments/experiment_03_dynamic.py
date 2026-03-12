"""
实验 3: 动态计算 - 重要Token选择
改进实验1，添加动态跳过机制
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
    hidden_dim = 256
    state_dim = 64
    max_seq_len = 128
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 10
    importance_threshold = 0.3  # 动态阈值
    skip_rate = 0.5  # 最多跳过50%的token
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


class DynamicStateLayer(nn.Module):
    """动态状态层 - 只处理重要token"""
    def __init__(self, hidden_dim, state_dim, dropout=0.1, threshold=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.threshold = threshold
        
        # 重要性预测器
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 状态层
        self.state_layer = LinearStateLayer(hidden_dim, state_dim, dropout)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 预测每个token的重要性
        importance = self.importance_net(x).squeeze(-1)  # [B, T]
        
        # 动态选择：重要token通过状态层，不重要的跳过
        selected_mask = importance > self.threshold  # [B, T]
        
        # 处理所有位置，但只对重要token更新状态
        output = torch.zeros_like(x)
        state = None
        
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]
            mask_t = selected_mask[:, t:t+1]  # [B, 1]
            
            if state is not None:
                out_t, state = self.state_layer(x_t, state)
            else:
                out_t, state = self.state_layer(x_t)
            
            # 重要token使用状态输出，不重要token直接使用输入
            output[:, t, :] = torch.where(
                mask_t.squeeze(-1).unsqueeze(-1),
                out_t.squeeze(1),
                x_t.squeeze(1)
            )
        
        output = self.norm(output + x)
        return output, importance


class DynamicDHSM(nn.Module):
    """动态DHSM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # 多层动态状态
        self.layers = nn.ModuleList([
            DynamicStateLayer(config.hidden_dim, config.state_dim, 
                             dropout=0.1, threshold=0.3)
            for _ in range(4)
        ])
        
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        
        total_importance = 0
        for layer in self.layers:
            x, importance = layer(x)
            total_importance += importance.mean().item()
        
        # 平均重要性
        avg_importance = total_importance / len(self.layers)
        
        return self.head(x), avg_importance, None


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    total_tokens = 0
    total_importance = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, importance, _ = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, importance, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        total_loss += loss.item() * inputs.numel()
        total_tokens += inputs.numel()
        total_importance += importance * inputs.size(0)
    
    avg_importance = total_importance / len(dataloader.dataset)
    return total_loss / total_tokens, avg_importance


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
    print("实验 3: 动态计算 - 重要Token选择")
    print("改进: 只对重要token进行完整状态更新")
    print("=" * 60)
    
    device = Config.device
    print(f"Device: {device}")
    print(f"Importance Threshold: {Config.importance_threshold}")
    
    model = DynamicDHSM(Config()).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")
    
    # 数据集
    train_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=50000)
    val_dataset = SimpleDataset(Config.vocab_size, Config.max_seq_len, num_samples=5000)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate * 0.5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.num_epochs)
    
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    best_val_loss = float('inf')
    
    print("\n[Training]")
    for epoch in range(Config.num_epochs):
        train_loss, importance = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"Epoch {epoch+1}/{Config.num_epochs} | NaN detected!")
            break
        
        print(f"Epoch {epoch+1}/{Config.num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Imp: {importance:.3f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/root/autodl-tmp/dhsm-research/experiments/best_model_exp3.pt')
    
    print("\n" + "=" * 60)
    print("实验 3 完成!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
