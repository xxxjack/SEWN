#!/usr/bin/env python3
"""
复杂任务验证实验
验证元认知层在复杂任务上的稳定性作用
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

# ==================== 配置 ====================

COMPLEX_CONFIG = {
    'num_train': 500,
    'num_val': 100,
    'max_seq_len': 4096,  # 长序列
    'vocab_size': 60000,  # 高词汇量
    'batch_size': 4,
    'hidden_dim': 512,    # 更大模型
    'lr': 5e-5,           # 更小学习率
    'epochs': 20,
    'patience': 5
}


# ==================== 复杂数据集 ====================

class ComplexMultiDocDataset(Dataset):
    """复杂多文档数据集"""
    
    def __init__(self, num_samples, max_seq_len, vocab_size):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        random.seed(42)
        self.data = []
        
        for _ in range(num_samples):
            # 模拟多文档输入 (2-5个文档)
            num_docs = random.randint(2, 5)
            docs = []
            
            for doc_idx in range(num_docs):
                doc_len = random.randint(500, 1000)
                # 高词汇量文档
                doc = torch.randint(1, vocab_size, (doc_len,))
                docs.append(doc)
            
            # 合并文档 (带分隔符)
            sep = torch.tensor([0])  # 分隔符
            input_ids = docs[0]
            for doc in docs[1:]:
                input_ids = torch.cat([input_ids, sep, doc])
            
            # 截断/填充
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
            else:
                padding = torch.zeros(max_seq_len - len(input_ids), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            
            # 目标: 压缩摘要
            target_len = max_seq_len // 8
            target = input_ids[:target_len]
            
            self.data.append({
                'input_ids': input_ids,
                'target': target,
                'num_docs': num_docs
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return {'input_ids': input_ids, 'targets': targets}


# ==================== 模型 ====================

class ComplexEncoder(nn.Module):
    """复杂任务编码器"""
    
    def __init__(self, vocab_size, hidden_dim, num_layers=4, num_heads=8):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 8192, hidden_dim) * 0.02
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) + self.pos_embedding[:, :seq_len, :]
        return self.encoder(x)


class MetaLite(nn.Module):
    """轻量元认知层"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.lr_adjuster = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        self.grad_adjuster = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        # 稳定性检测器
        self.stability_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.base_lr = 5e-5
        self.max_grad = 1.0
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return {
            'lr_factor': self.lr_adjuster(pooled),
            'grad_clip': self.max_grad * self.grad_adjuster(pooled),
            'stability_score': self.stability_detector(pooled)
        }


class FullMetacognition(nn.Module):
    """完整元认知层"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # 状态评估器
        self.state_eval = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        # 学习率调节
        self.lr_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 梯度调节
        self.grad_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 置信度评估
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 损失缩放
        self.loss_scale = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.base_lr = 5e-5
        self.max_grad = 1.0
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        state = self.state_eval(pooled)
        
        return {
            'state': state,
            'lr_factor': self.lr_gate(state),
            'grad_clip': self.max_grad * self.grad_gate(state),
            'confidence': self.confidence(state),
            'loss_scale': 0.5 + 0.5 * self.loss_scale(state)
        }


class ModelA_NoMeta(nn.Module):
    """组A: 无元认知层"""
    
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.encoder = ComplexEncoder(vocab_size, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "A_NoMeta"
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        logits = self.decoder(hidden)
        return logits, {}


class ModelB_FullMeta(nn.Module):
    """组B: 完整元认知层"""
    
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.encoder = ComplexEncoder(vocab_size, hidden_dim)
        self.metacognition = FullMetacognition(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "B_FullMeta"
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        meta = self.metacognition(hidden)
        logits = self.decoder(hidden)
        return logits, meta


class ModelC_MetaLite(nn.Module):
    """组C: MetaLite"""
    
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.encoder = ComplexEncoder(vocab_size, hidden_dim)
        self.metacognition = MetaLite(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "C_MetaLite"
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        meta = self.metacognition(hidden)
        logits = self.decoder(hidden)
        return logits, meta


class ModelD_External(nn.Module):
    """组D: 外部调度"""
    
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.encoder = ComplexEncoder(vocab_size, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        self.name = "D_External"
        
        self.register_buffer('step_count', torch.tensor(0))
        self.warmup_steps = 200
    
    def forward(self, input_ids):
        hidden = self.encoder(input_ids)
        logits = self.decoder(hidden)
        
        step = self.step_count.item()
        if step < self.warmup_steps:
            lr_factor = step / self.warmup_steps
        else:
            # 余弦退火
            progress = (step - self.warmup_steps) / 1000
            lr_factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        meta = {
            'lr_factor': torch.tensor([[lr_factor]] * input_ids.size(0)),
            'grad_clip': torch.tensor([[1.0]] * input_ids.size(0))
        }
        
        return logits, meta
    
    def step(self):
        self.step_count += 1


# ==================== 训练 ====================

def train_epoch(model, dataloader, optimizer, device, config, model_D=False):
    model.train()
    total_loss = 0
    num_batches = 0
    nan_detected = False
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        logits, meta = model(input_ids)
        
        # 计算loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        
        min_len = min(shift_logits.size(1), shift_labels.size(1))
        shift_logits = shift_logits[:, :min_len, :]
        shift_labels = shift_labels[:, :min_len]
        
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=0
        )
        
        # 检查NaN
        if torch.isnan(loss):
            print("  ⚠️ NaN detected in loss!")
            nan_detected = True
            return {'loss': float('nan'), 'nan_detected': True}
        
        # 元认知调节
        if 'loss_scale' in meta:
            loss = loss * meta['loss_scale'].mean()
        
        loss.backward()
        
        # 梯度裁剪
        if 'grad_clip' in meta:
            clip_val = max(meta['grad_clip'].mean().item(), 0.1)
        else:
            clip_val = 1.0
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        
        optimizer.step()
        
        if model_D:
            model.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'nan_detected': nan_detected
    }


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            logits, _ = model(input_ids)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            
            min_len = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]
            
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=0
            )
            
            if torch.isnan(loss):
                return {'loss': float('nan'), 'nan_detected': True}
            
            total_loss += loss.item()
            num_batches += 1
    
    return {'loss': total_loss / num_batches}


def run_experiment(model_class, name, config, device):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"{'='*60}")
    
    # 数据
    train_data = ComplexMultiDocDataset(
        config['num_train'], config['max_seq_len'], config['vocab_size']
    )
    val_data = ComplexMultiDocDataset(
        config['num_val'], config['max_seq_len'], config['vocab_size']
    )
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn=collate_fn)
    
    # 模型
    model = model_class(config['vocab_size'], config['hidden_dim']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    collapsed = False
    
    for epoch in range(config['epochs']):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, name == "D_External")
        val_metrics = evaluate(model, val_loader, device)
        
        # 检查崩溃
        if math.isnan(train_metrics['loss']) or math.isnan(val_metrics['loss']):
            print(f"Epoch {epoch+1}: ⚠️ TRAINING COLLAPSED!")
            collapsed = True
            break
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        })
        
        print(f"Epoch {epoch+1}: train={train_metrics['loss']:.4f}, val={val_metrics['loss']:.4f}")
        
        # 早停
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return {
        'name': name,
        'params': total_params,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1,
        'collapsed': collapsed,
        'stable': not collapsed,
        'history': history
    }


def main():
    print("="*60)
    print("  复杂任务验证实验")
    print("="*60)
    print(f"Config: {COMPLEX_CONFIG}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    experiments = [
        (ModelA_NoMeta, "A_NoMeta"),
        (ModelB_FullMeta, "B_FullMeta"),
        (ModelC_MetaLite, "C_MetaLite"),
        (ModelD_External, "D_External"),
    ]
    
    results = []
    for model_class, name in experiments:
        try:
            result = run_experiment(model_class, name, COMPLEX_CONFIG, device)
            results.append(result)
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append({'name': name, 'error': str(e), 'stable': False, 'collapsed': True})
    
    # 结果
    print("\n" + "="*60)
    print("  复杂任务实验结果")
    print("="*60)
    print(f"{'Model':<15} {'Params':>10} {'Best Val':>12} {'Epochs':>8} {'Stable':>8}")
    print("-"*60)
    
    for r in results:
        if 'error' not in r:
            status = '✅' if r['stable'] else '❌ 崩溃'
            print(f"{r['name']:<15} {r['params']:>10,} {r['best_val_loss']:>12.4f} {r['final_epoch']:>8} {status:>8}")
        else:
            print(f"{r['name']:<15} {'ERROR':>10} {'-':>12} {'-':>8} {'❌':>8}")
    
    # 保存
    output = {
        'config': COMPLEX_CONFIG,
        'results': [{k: v for k, v in r.items() if k != 'history'} for r in results],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('complex_task_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存到 complex_task_results.json")
    
    return results


if __name__ == "__main__":
    main()
