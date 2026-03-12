"""
实验8: 多世界SSM (Multi-World SSM) - SEWN第一阶段
海蓝 🌊 - 2026-03-12

核心创新: 4个并行状态空间 + 动态路由
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import numpy as np
import math
import time
from pathlib import Path
import sys

# ==================== 配置 ====================
CONFIG = {
    'device': 'cuda',
    'max_seq_length': 256,
    'vocab_size': 10000,
    'hidden_dim': 512,
    'state_dims': [64, 128, 256, 512],  # 4个世界的状态维度
    'num_layers': 4,
    'dropout': 0.1,
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'data_path': '/root/autodl-tmp/dhsm-research/data/wikitext2',
    'save_path': '/root/autodl-tmp/dhsm-research/experiments/',
    'log_interval': 50,
}

# ==================== 进度显示 ====================
class ProgressTracker:
    def __init__(self, total_steps, desc="训练"):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.desc = desc
    
    def update(self, step, loss=None, extra_info=None):
        self.current_step = step
        progress = step / self.total_steps * 100
        
        elapsed = time.time() - self.start_time
        if step > 0:
            eta = elapsed / step * (self.total_steps - step)
            eta_str = f"ETA: {int(eta//60)}m{int(eta%60)}s"
        else:
            eta_str = "ETA: --"
        
        loss_str = f"Loss: {loss:.4f}" if loss else ""
        extra_str = f" | {extra_info}" if extra_info else ""
        
        bar = self._make_bar(progress)
        
        msg = f"\r[{self.desc}] {bar} {progress:.1f}% ({step}/{self.total_steps}) | {loss_str} | {elapsed:.0f}s | {eta_str}{extra_str}"
        print(msg, end='', flush=True)
        
        if step >= self.total_steps:
            print()  # 换行
    
    def _make_bar(self, progress, width=30):
        filled = int(width * progress / 100)
        empty = width - filled
        return '█' * filled + '░' * empty
    
    def epoch_summary(self, epoch, train_loss, val_loss, ppl):
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} 完成!")
        print(f"  训练Loss: {train_loss:.4f}")
        print(f"  验证Loss: {val_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  耗时: {elapsed/60:.1f}分钟")
        print(f"{'='*60}\n")

# ==================== 单世界SSM ====================
class SSMWorld(nn.Module):
    """单个世界状态空间层"""
    def __init__(self, input_dim, state_dim, name="world"):
        super().__init__()
        self.name = name
        self.state_dim = state_dim
        
        # SSM参数
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, input_dim, bias=False)
        self.D = nn.Linear(input_dim, input_dim, bias=False)
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(CONFIG['dropout'])
    
    def forward(self, x):
        residual = x
        batch_size, seq_len, _ = x.shape
        
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            state = torch.matmul(state, self.A.T) + self.B(x_t)
            state = torch.clamp(state, -10, 10)
            y_t = self.C(state) + self.D(x_t)
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        output = self.layer_norm(output + residual)
        output = self.dropout(output)
        return output

# ==================== 多世界SSM层 ====================
class MultiWorldSSM(nn.Module):
    """多世界并行SSM + 动态路由"""
    def __init__(self, input_dim, state_dims):
        super().__init__()
        self.num_worlds = len(state_dims)
        
        # 创建多个世界
        self.worlds = nn.ModuleList([
            SSMWorld(input_dim, dim, name=f"world_{i}_dim{dim}")
            for i, dim in enumerate(state_dims)
        ])
        
        # 世界路由器
        self.router = nn.Linear(input_dim, self.num_worlds)
        
        # 世界使用统计
        self.register_buffer('world_usage', torch.zeros(self.num_worlds))
    
    def forward(self, x):
        # 计算路由权重
        route_logits = self.router(x)  # (batch, seq, num_worlds)
        routes = F.softmax(route_logits, dim=-1)  # 归一化
        
        # 记录使用情况
        self.world_usage += routes.mean(dim=(0, 1)).detach()
        
        # 每个世界处理
        world_outputs = []
        for i, world in enumerate(self.worlds):
            out = world(x)
            world_outputs.append(out)
        
        # 加权融合
        output = torch.zeros_like(x)
        for i, out in enumerate(world_outputs):
            weight = routes[..., i].unsqueeze(-1)  # (batch, seq, 1)
            output = output + weight * out
        
        return output, routes

# ==================== 完整模型 ====================
class SEWNModel(nn.Module):
    """SEWN第一阶段: 多世界SSM语言模型"""
    def __init__(self, vocab_size, hidden_dim, state_dims, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(CONFIG['max_seq_length'], hidden_dim)
        
        # 多层多世界SSM
        self.layers = nn.ModuleList([
            MultiWorldSSM(hidden_dim, state_dims)
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
        
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        all_routes = []
        for layer in self.layers:
            x, routes = layer(x)
            all_routes.append(routes)
        
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits, 'routes': all_routes}
    
    def get_world_usage(self):
        """获取各世界使用统计"""
        usage = []
        for layer in self.layers:
            usage.append(layer.world_usage.clone())
        return torch.stack(usage).mean(dim=0)

# ==================== 数据集 ====================
class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.examples = []
        print(f"[数据预处理] 开始处理 {len(texts)} 条文本...")
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                pct = i / len(texts) * 100
                print(f"\r  进度: {pct:.1f}% ({i}/{len(texts)})", end='', flush=True)
            
            if len(text.strip()) < 10:
                continue
            tokens = tokenizer.encode(text).ids
            if len(tokens) < 2:
                continue
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            self.examples.append(tokens)
        
        print(f"\r  进度: 100.0% ({len(texts)}/{len(texts)}) ✓")
        print(f"[数据预处理] 完成! 有效样本: {len(self.examples)}")
    
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return {'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
                'labels': torch.tensor(tokens[1:], dtype=torch.long)}

# ==================== 训练函数 ====================
def train():
    print("=" * 60)
    print("实验8: 多世界SSM (Multi-World SSM) - SEWN第一阶段")
    print("=" * 60)
    
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    
    # 加载数据
    print("\n[加载数据集]")
    dataset = load_from_disk(CONFIG['data_path'])
    print(f"训练样本: {len(dataset['train'])}")
    print(f"验证样本: {len(dataset['validation'])}")
    
    # 加载tokenizer
    print("\n[加载Tokenizer]")
    tokenizer_path = Path(CONFIG['save_path']) / "tokenizer_wikitext.json"
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    # 准备数据
    print("\n[准备训练数据]")
    train_texts = [item['text'] for item in dataset['train']]
    valid_texts = [item['text'] for item in dataset['validation']]
    
    train_dataset = WikiTextDataset(train_texts, tokenizer, CONFIG['max_seq_length'])
    valid_dataset = WikiTextDataset(valid_texts, tokenizer, CONFIG['max_seq_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                              shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'],
                              num_workers=2, pin_memory=True)
    
    print(f"\n训练批次: {len(train_loader)}")
    print(f"验证批次: {len(valid_loader)}")
    
    # 创建模型
    print("\n[创建模型]")
    model = SEWNModel(
        CONFIG['vocab_size'],
        CONFIG['hidden_dim'],
        CONFIG['state_dims'],
        CONFIG['num_layers']
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {num_params/1e6:.2f}M")
    print(f"世界配置: {CONFIG['state_dims']}")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs']
    )
    
    # 训练
    print("\n[开始训练]")
    print("=" * 60)
    
    best_val_loss = float('inf')
    total_start = time.time()
    
    for epoch in range(CONFIG['num_epochs']):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_losses = []
        progress = ProgressTracker(len(train_loader), f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
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
            progress.update(batch_idx + 1, loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证
        model.eval()
        val_losses = []
        
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, labels)
                val_losses.append(outputs['loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        ppl = math.exp(avg_val_loss)
        
        # 打印epoch总结
        epoch_time = time.time() - epoch_start
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} 完成!")
        print(f"  训练Loss: {avg_train_loss:.4f}")
        print(f"  验证Loss: {avg_val_loss:.4f}")
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  耗时: {epoch_time/60:.1f}分钟")
        
        # 世界使用情况
        world_usage = model.get_world_usage()
        print(f"  世界使用分布: {world_usage.tolist()}")
        print(f"{'='*60}\n")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_file = Path(CONFIG['save_path']) / "best_model_exp8.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
                'ppl': ppl,
                'config': CONFIG,
            }, save_file)
            print(f"  ✓ 保存最佳模型 (Val: {best_val_loss:.4f}, PPL: {ppl:.2f})")
        
        scheduler.step()
    
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("实验8 完成!")
    print(f"最佳验证Loss: {best_val_loss:.4f}")
    print(f"最佳Perplexity: {math.exp(best_val_loss):.2f}")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print("=" * 60)

if __name__ == '__main__':
    train()
