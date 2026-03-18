#!/usr/bin/env python3
"""
Exp10-MetaLite 训练脚本
验证"元认知层 = 训练稳定器"假说
"""

import os
import sys
import json
import random
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 添加 ASN 模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asn import ComplexityEstimator, ComplexityLevel


# ==================== 数据集 ====================

class SyntheticMultiNewsDataset(Dataset):
    """合成多文档摘要数据集"""
    
    def __init__(self, num_samples=1000, max_seq_len=512, vocab_size=10000):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # 预生成数据
        random.seed(42)
        self.data = []
        for _ in range(num_samples):
            seq_len = random.randint(128, max_seq_len)
            input_ids = torch.randint(1, vocab_size, (seq_len,))
            
            # 简单目标: 输入的压缩版本
            target_len = seq_len // 4
            target = input_ids[:target_len]
            
            self.data.append({
                'input_ids': input_ids,
                'target': target,
                'seq_len': seq_len
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """动态padding"""
    input_ids_list = [item['input_ids'] for item in batch]
    target_list = [item['target'] for item in batch]
    
    # Pad inputs
    max_input_len = max(len(x) for x in input_ids_list)
    input_ids = torch.zeros(len(batch), max_input_len, dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        input_ids[i, :len(ids)] = ids
    
    # Pad targets
    max_target_len = max(len(x) for x in target_list)
    targets = torch.zeros(len(batch), max_target_len, dtype=torch.long)
    for i, t in enumerate(target_list):
        targets[i, :len(t)] = t
    
    return {'input_ids': input_ids, 'targets': targets}


# ==================== 模型定义 ====================

class SimpleTransformer(nn.Module):
    """简单Transformer编码器"""
    
    def __init__(self, vocab_size=10000, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        return self.encoder(x)


class MetaLite(nn.Module):
    """轻量元认知层 - 只保留核心稳定功能"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 简化: 只做学习率调节
        self.lr_adjuster = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 梯度裁剪调节
        self.grad_clip_adjuster = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.base_lr = 1e-4
        self.max_grad_clip = 1.0
    
    def forward(self, hidden_states):
        """返回调节因子"""
        pooled = hidden_states.mean(dim=1)
        
        lr_factor = self.lr_adjuster(pooled)
        grad_factor = self.grad_clip_adjuster(pooled)
        
        return {
            'lr_factor': lr_factor,
            'grad_clip': self.max_grad_clip * grad_factor
        }


class FullMetacognition(nn.Module):
    """完整元认知层"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 完整版: 包含状态评估 + 学习率调节 + 梯度裁剪 + 世界门控
        self.state_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        self.lr_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.grad_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.confidence_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.base_lr = 1e-4
        self.max_grad_clip = 1.0
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        state = self.state_evaluator(pooled)
        
        return {
            'state': state,
            'lr_factor': self.lr_gate(state),
            'grad_clip': self.max_grad_clip * self.grad_gate(state),
            'confidence': self.confidence_gate(state)
        }


class ModelA(nn.Module):
    """组A: 无元认知层"""
    
    def __init__(self, vocab_size=10000, hidden_dim=256):
        super().__init__()
        self.encoder = SimpleTransformer(vocab_size, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "A_NoMeta"
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        logits = self.decoder(hidden)
        return logits, {}


class ModelB(nn.Module):
    """组B: 完整元认知层"""
    
    def __init__(self, vocab_size=10000, hidden_dim=256):
        super().__init__()
        self.encoder = SimpleTransformer(vocab_size, hidden_dim)
        self.metacognition = FullMetacognition(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "B_FullMeta"
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        meta = self.metacognition(hidden)
        logits = self.decoder(hidden)
        return logits, meta


class ModelC(nn.Module):
    """组C: MetaLite (轻量元认知)"""
    
    def __init__(self, vocab_size=10000, hidden_dim=256):
        super().__init__()
        self.encoder = SimpleTransformer(vocab_size, hidden_dim)
        self.metacognition = MetaLite(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "C_MetaLite"
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        meta = self.metacognition(hidden)
        logits = self.decoder(hidden)
        return logits, meta


class ModelD(nn.Module):
    """组D: 外部调度 (无内部元认知, 使用外部规则)"""
    
    def __init__(self, vocab_size=10000, hidden_dim=256):
        super().__init__()
        self.encoder = SimpleTransformer(vocab_size, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "D_External"
        
        # 外部调度参数
        self.register_buffer('step_count', torch.tensor(0))
        self.warmup_steps = 100
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        logits = self.decoder(hidden)
        
        # 外部调度逻辑
        step = self.step_count.item()
        if step < self.warmup_steps:
            lr_factor = step / self.warmup_steps
        else:
            lr_factor = 1.0
        
        meta = {
            'lr_factor': torch.tensor([[lr_factor]] * input_ids.size(0)),
            'grad_clip': torch.tensor([[1.0]] * input_ids.size(0))
        }
        
        return logits, meta
    
    def step(self):
        self.step_count += 1


# ==================== 训练函数 ====================

def train_epoch(model, dataloader, optimizer, device, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_meta_lr = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        logits, meta = model(input_ids)
        
        # 计算loss (简单的next token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous() if targets.size(1) > 1 else targets
        
        # 确保维度匹配
        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]
        
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=0
        )
        
        loss.backward()
        
        # 元认知调节梯度裁剪
        if 'grad_clip' in meta:
            clip_value = meta['grad_clip'].mean().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max(clip_value, 0.1))
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # 外部调度步进
        if hasattr(model, 'step'):
            model.step()
        
        total_loss += loss.item()
        if 'lr_factor' in meta:
            total_meta_lr += meta['lr_factor'].mean().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'avg_lr_factor': total_meta_lr / len(dataloader) if total_meta_lr > 0 else 1.0
    }


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            logits, _ = model(input_ids)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous() if targets.size(1) > 1 else targets
            
            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]
            
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=0
            )
            
            total_loss += loss.item()
    
    return {'loss': total_loss / len(dataloader)}


def run_experiment(model_class, model_name, config, device):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"  Experiment: {model_name}")
    print(f"{'='*60}")
    
    # 数据
    train_dataset = SyntheticMultiNewsDataset(
        num_samples=config['num_train'],
        max_seq_len=config['max_seq_len']
    )
    val_dataset = SyntheticMultiNewsDataset(
        num_samples=config['num_val'],
        max_seq_len=config['max_seq_len']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        collate_fn=collate_fn
    )
    
    # 模型
    model = model_class(
        vocab_size=10000,
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # 训练
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    
    for epoch in range(config['epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'lr_factor': train_metrics['avg_lr_factor']
        })
        
        print(f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, "
              f"lr_factor={train_metrics['avg_lr_factor']:.3f}")
        
        # 早停检查
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # NaN检查
        if math.isnan(train_metrics['loss']) or math.isnan(val_metrics['loss']):
            print(f"⚠️ NaN detected at epoch {epoch+1}! Training collapsed!")
            break
    
    return {
        'model_name': model_name,
        'total_params': total_params,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'history': history,
        'stable': not math.isnan(best_val_loss)
    }


def main():
    """主函数"""
    config = {
        'num_train': 1000,
        'num_val': 200,
        'max_seq_len': 512,
        'batch_size': 16,
        'hidden_dim': 256,
        'lr': 1e-4,
        'epochs': 10,
        'patience': 3
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: {config}")
    
    # 运行4组实验
    experiments = [
        (ModelA, "A_NoMeta"),
        (ModelB, "B_FullMeta"),
        (ModelC, "C_MetaLite"),
        (ModelD, "D_External"),
    ]
    
    results = []
    for model_class, name in experiments:
        try:
            result = run_experiment(model_class, name, config, device)
            results.append(result)
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append({
                'model_name': name,
                'error': str(e),
                'stable': False
            })
    
    # 输出对比结果
    print("\n" + "="*60)
    print("  实验结果对比")
    print("="*60)
    print(f"{'Model':<15} {'Params':>10} {'Best Val Loss':>15} {'Epochs':>8} {'Stable':>8}")
    print("-"*60)
    for r in results:
        if 'error' not in r:
            print(f"{r['model_name']:<15} {r['total_params']:>10,} "
                  f"{r['best_val_loss']:>15.4f} {r['final_epoch']:>8} "
                  f"{'✅' if r['stable'] else '❌':>8}")
        else:
            print(f"{r['model_name']:<15} {'ERROR':>10} {'-':>15} {'-':>8} {'❌':>8}")
    
    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'exp10_metalite_results.json')
    
    # 转换为可序列化格式
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'history'}
        if 'history' in r:
            sr['history'] = r['history']
        serializable_results.append(sr)
    
    with open(output_file, 'w') as f:
        json.dump({
            'config': config,
            'results': serializable_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    main()
