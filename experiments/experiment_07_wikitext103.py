"""
实验7: DHSM WikiText-103 大规模训练
海蓝 🌊 - 2026-03-12
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import numpy as np
import math
from pathlib import Path
import time

CONFIG = {
    'device': 'cuda',
    'max_seq_length': 256,
    'vocab_size': 10000,
    'hidden_dim': 768,
    'state_dim': 128,
    'num_layers': 6,
    'dropout': 0.1,
    'batch_size': 128,
    'num_epochs': 3,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'data_path': '/root/autodl-tmp/dhsm-research/data/wikitext103',
    'save_path': '/root/autodl-tmp/dhsm-research/experiments/',
    'log_interval': 200,
}

class StateSpaceLayer(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, input_dim, bias=False)
        self.D = nn.Linear(input_dim, input_dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(input_dim, 2), nn.Softmax(dim=-1))
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
        gate = self.gate(x)
        output = gate[:, :, 0:1] * output + gate[:, :, 1:2] * residual
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output

class DHSMLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, state_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(CONFIG['max_seq_length'], hidden_dim)
        self.layers = nn.ModuleList([StateSpaceLayer(hidden_dim, state_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.embedding.weight, 0, 0.02)
        nn.init.normal_(self.pos_embedding.weight, 0, 0.02)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {'loss': loss, 'logits': logits}

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.examples = []
        for text in texts:
            if len(text.strip()) < 10: continue
            tokens = tokenizer.encode(text).ids
            if len(tokens) < 2: continue
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [0] * (max_length - len(tokens))
            self.examples.append(tokens)
    
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return {'input_ids': torch.tensor(tokens[:-1], dtype=torch.long), 'labels': torch.tensor(tokens[1:], dtype=torch.long)}

def train():
    print("=" * 60)
    print("实验7: DHSM WikiText-103 大规模训练")
    print("=" * 60)
    device = torch.device(CONFIG['device'])
    print(f"Device: {device}")
    print("\n[Loading Dataset]")
    dataset = load_from_disk(CONFIG['data_path'])
    print(f"Train: {len(dataset['train'])} samples")
    print(f"Valid: {len(dataset['validation'])} samples")
    print("\n[Loading Tokenizer]")
    tokenizer_path = Path(CONFIG['save_path']) / "tokenizer_wikitext.json"
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print("\n[Preparing Data]")
    train_texts = [item['text'] for item in dataset['train']]
    valid_texts = [item['text'] for item in dataset['validation']]
    train_dataset = WikiTextDataset(train_texts, tokenizer, CONFIG['max_seq_length'])
    valid_dataset = WikiTextDataset(valid_texts, tokenizer, CONFIG['max_seq_length'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], num_workers=2, pin_memory=True)
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print("\n[Creating Model]")
    model = DHSMLanguageModel(CONFIG['vocab_size'], CONFIG['hidden_dim'], CONFIG['state_dim'], CONFIG['num_layers'])
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    print("\n[Training]")
    best_val_loss = float('inf')
    start_time = time.time()
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        train_losses = []
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
            if batch_idx % CONFIG['log_interval'] == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Time: {elapsed/60:.1f}min")
        avg_train_loss = np.mean(train_losses)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, labels)
                val_losses.append(outputs['loss'].item())
        avg_val_loss = np.mean(val_losses)
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Perplexity: {math.exp(avg_val_loss):.2f}")
        print(f"  Time: {elapsed/60:.1f}min")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'val_loss': best_val_loss, 'config': CONFIG}, Path(CONFIG['save_path']) / "best_model_exp7.pt")
            print(f"  ✓ Saved best model (Val: {best_val_loss:.4f})")
        scheduler.step()
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("实验7 完成!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Perplexity: {math.exp(best_val_loss):.2f}")
    print(f"Total Time: {total_time/60:.1f}min")
    print("=" * 60)

if __name__ == '__main__':
    train()
