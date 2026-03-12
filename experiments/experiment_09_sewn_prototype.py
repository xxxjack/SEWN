#!/usr/bin/env python3
"""
实验9 - SEWN原型 v0.1
添加意识层 + 元认知层 + 动态路由v2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from tokenizers import Tokenizer
import math
import time

CONFIG = {
    'device': 'cuda',
    'max_seq_length': 256,
    'vocab_size': 10000,
    'hidden_dim': 512,
    'state_dims': [64, 128, 256, 512],
    'num_layers': 4,
    'dropout': 0.1,
    'batch_size': 64,
    'num_epochs': 10,
    'learning_rate': 8e-5,
    'weight_decay': 0.01,
    'save_path': '/root/autodl-tmp/dhsm-research/experiments/',
}

class SSMWorld(nn.Module):
    def __init__(self, hidden_dim, state_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Linear(hidden_dim, state_dim)
        self.C = nn.Linear(state_dim, hidden_dim)
        self.D = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(CONFIG['dropout'])
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        dt = 0.1
        A_discrete = torch.matrix_exp(self.A * dt)
        h = torch.zeros(batch, self.state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            h = torch.matmul(h, A_discrete.t()) + self.B(x[:, t])
            y = self.C(h) + self.D(x[:, t])
            outputs.append(y)
        output = torch.stack(outputs, dim=1)
        return self.dropout(self.layer_norm(output + x))

class ConsciousMultiWorldSSM(nn.Module):
    def __init__(self, hidden_dim, state_dims):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_worlds = len(state_dims)
        self.worlds = nn.ModuleList([SSMWorld(hidden_dim, sd) for sd in state_dims])
        self.router = nn.Linear(hidden_dim, self.n_worlds)
        self.consciousness = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.conscious_norm = nn.LayerNorm(hidden_dim)
        self.metacognition = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.register_buffer('world_usage', torch.zeros(self.n_worlds))
        
    def forward(self, x, return_consciousness=False):
        batch, seq_len, _ = x.shape
        conscious_out, attn_weights = self.consciousness(x, x, x)
        x = self.conscious_norm(x + conscious_out)
        confidence = self.metacognition(x.mean(dim=1))
        base_routes = self.router(x.mean(dim=1))
        route_temp = 1.0 + (1.0 - confidence) * 2.0
        routes = F.softmax(base_routes / route_temp, dim=-1)
        self.world_usage += routes.sum(dim=0).detach()
        
        # 修复: 先计算所有世界输出，再加权组合
        world_outputs = [world(x) for world in self.worlds]
        output = torch.zeros_like(x)
        for i, w_out in enumerate(world_outputs):
            output = output + routes[:, i].unsqueeze(1).unsqueeze(2) * w_out
        
        if return_consciousness:
            return output, attn_weights, confidence, routes
        return output, routes

class SEWNModelV2(nn.Module):
    def __init__(self, vocab_size, hidden_dim, state_dims, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(CONFIG['max_seq_length'], hidden_dim)
        self.layers = nn.ModuleList([ConsciousMultiWorldSSM(hidden_dim, state_dims) for _ in range(num_layers)])
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
        all_routes, all_confidence = [], []
        for layer in self.layers:
            x, routes = layer(x)
            all_routes.append(routes)
            all_confidence.append(layer.metacognition(x.mean(dim=1)).mean().item())
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).float()
            correct_mask = correct.mean(dim=-1)
            confidence_tensor = torch.tensor(all_confidence, device=logits.device).mean()
            meta_loss = F.mse_loss(confidence_tensor, correct_mask.mean())
            loss = loss + 0.1 * meta_loss
        return {'loss': loss, 'logits': logits, 'routes': all_routes, 'confidence': sum(all_confidence)/len(all_confidence)}
    
    def get_world_usage(self):
        return torch.stack([layer.world_usage for layer in self.layers]).mean(dim=0)

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.examples = []
        for text in texts:
            if len(text.strip()) < 10: continue
            tokens = tokenizer.encode(text).ids
            if len(tokens) < 2: continue
            if len(tokens) > max_length: tokens = tokens[:max_length]
            else: tokens = tokens + [0] * (max_length - len(tokens))
            self.examples.append(tokens)
        print(f'[数据集] 有效样本: {len(self.examples)}')
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        x = torch.tensor(self.examples[idx], dtype=torch.long)
        return x, x.clone()

def train():
    print('=' * 60)
    print('实验9 - SEWN原型 v0.1 (意识+元认知)')
    print('=' * 60)
    device = torch.device(CONFIG['device'])
    print('加载WikiText-103...')
    dataset = load_from_disk('/root/autodl-tmp/dhsm-research/data/wikitext103')
    tokenizer = Tokenizer.from_file('/root/autodl-tmp/dhsm-research/experiments/tokenizer_wikitext.json')
    train_dataset = WikiTextDataset(dataset['train']['text'], tokenizer, CONFIG['max_seq_length'])
    val_dataset = WikiTextDataset(dataset['validation']['text'], tokenizer, CONFIG['max_seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], num_workers=4)
    model = SEWNModelV2(CONFIG['vocab_size'], CONFIG['hidden_dim'], CONFIG['state_dims'], CONFIG['num_layers']).to(device)
    try:
        ckpt = torch.load(CONFIG['save_path'] + 'best_model_exp8.pt', map_location=device, weights_only=False)
        model_dict = model.state_dict()
        matched = {k: v for k, v in ckpt['model_state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f'加载实验8权重: {len(matched)}/{len(model_dict)} 参数')
    except Exception as e:
        print(f'从实验8加载失败: {e}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    best_ppl = float('inf')
    start_time = time.time()
    log_file = CONFIG['save_path'] + 'exp09_output.log'
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        model.train()
        train_loss = 0
        epoch_start = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, y)
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f'[Epoch {epoch}/{CONFIG["num_epochs"]}] {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Conf: {output["confidence"]:.3f}', end='\r')
        model.eval()
        val_loss = 0
        avg_confidence = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x, y)
                val_loss += output['loss'].item()
                avg_confidence += output['confidence']
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        ppl = math.exp(min(val_loss, 10))
        avg_confidence /= len(val_loader)
        epoch_time = (time.time() - epoch_start) / 60
        world_usage = model.get_world_usage()
        print(f'\nEpoch {epoch} 完成! Val Loss: {val_loss:.4f} | PPL: {ppl:.2f} | Conf: {avg_confidence:.3f} | 耗时: {epoch_time:.1f}分钟')
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_ppl': ppl}, CONFIG['save_path'] + 'best_model_exp9.pt')
            print(f'  保存最佳模型 (PPL: {ppl:.2f})')
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch}: Val PPL={ppl:.2f}, Conf={avg_confidence:.3f}\n')
    total_time = (time.time() - start_time) / 60
    print('=' * 60)
    print(f'实验9 完成! 最佳PPL: {best_ppl:.2f} | 总耗时: {total_time:.1f}分钟')
    print('=' * 60)

if __name__ == '__main__':
    train()
