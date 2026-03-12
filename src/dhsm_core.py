"""
DHSM - Dynamic Hierarchical State Machine
动态层级状态机模型

核心架构:
1. LinearStateLayer - 线性状态层 (SSM风格)
2. DynamicDepthRouter - 动态深度路由
3. HierarchicalMemory - 分层记忆系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DHSMConfig:
    """DHSM 模型配置"""
    vocab_size: int = 50257
    hidden_dim: int = 512
    num_layers: int = 6
    state_dim: int = 128
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512
    
    # DHSM 特有参数
    dynamic_depth: bool = False
    min_depth: int = 4
    memory_levels: int = 3
    state_compression: float = 0.5


class LinearStateLayer(nn.Module):
    """线性状态层 - SSM 风格"""
    
    def __init__(self, hidden_dim: int, state_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        self.A_log = nn.Parameter(torch.randn(state_dim))
        self.B = nn.Linear(hidden_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, hidden_dim, bias=False)
        self.D = nn.Parameter(torch.randn(hidden_dim))
        
        self.select_gate = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        with torch.no_grad():
            self.A_log.data = torch.log(torch.exp(-0.5 * torch.ones(self.state_dim)) + 0.1 * torch.randn(self.state_dim))
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        A = torch.exp(self.A_log)
        
        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=device)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            gate = self.select_gate(x_t)
            Bx = self.B(x_t)
            state = A * state + gate * Bx
            y_t = self.C(state) + self.D * x_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)
        output = self.norm(output + x)
        output = self.dropout(self.out_proj(output))
        
        return output, state


class DynamicDepthRouter(nn.Module):
    """动态深度路由器"""
    
    def __init__(self, hidden_dim: int, num_layers: int, min_depth: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_depth = min_depth
        
        self.depth_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_layers - min_depth + 1),
            nn.Softmax(dim=-1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, int]:
        pooled = x.mean(dim=1)
        depth_logits = self.depth_predictor(pooled).mean(dim=0)
        
        if training:
            depth_weights = F.gumbel_softmax(depth_logits * self.temperature, hard=False)
            selected_depth = torch.multinomial(depth_weights, 1).item() + self.min_depth
        else:
            depth_weights = F.softmax(depth_logits / self.temperature, dim=-1)
            selected_depth = torch.argmax(depth_weights).item() + self.min_depth
        
        return depth_weights, selected_depth


class HierarchicalMemory(nn.Module):
    """分层记忆系统"""
    
    def __init__(self, hidden_dim: int, state_dim: int, num_levels: int = 3, compression_ratio: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_levels = num_levels
        
        self.capacities = [256, 1024, 4096]
        
        self.compressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, max(32, int(hidden_dim * compression_ratio ** i))),
                nn.ReLU(),
                nn.Linear(max(32, int(hidden_dim * compression_ratio ** i)), hidden_dim)
            ) for i in range(num_levels)
        ])
        
        self.retriever = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor, memory_bank: Optional[List[torch.Tensor]] = None):
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if memory_bank is None:
            memory_bank = [torch.zeros(batch_size, 0, self.hidden_dim, device=device) 
                          for _ in range(self.num_levels)]
        
        retrieved = []
        for level, (mem, compressor) in enumerate(zip(memory_bank, self.compressors)):
            if mem.shape[1] > 0:
                compressed = compressor(mem[:, -min(256, mem.shape[1]):, :])
                key = compressed.expand(batch_size, -1, -1)
                value = key
                retrieved_info, _ = self.retriever(x, key, value)
                retrieved.append(retrieved_info)
        
        if retrieved:
            combined = sum(retrieved) / len(retrieved)
            gate_input = torch.cat([x, combined], dim=-1)
            gate_weight = torch.sigmoid(self.gate(gate_input))
            enhanced = gate_weight * x + (1 - gate_weight) * combined
        else:
            enhanced = x
        
        # 更新记忆
        updated_memory = []
        last_mem = x[:, -1:, :]  # 取最后一个token
        for level in range(self.num_levels):
            old_mem = memory_bank[level]
            new_mem = torch.cat([old_mem, last_mem], dim=1)
            capacity = self.capacities[level]
            if new_mem.shape[1] > capacity:
                new_mem = new_mem[:, -capacity:, :]
            updated_memory.append(new_mem)
        
        return enhanced, updated_memory


class DHSMBlock(nn.Module):
    """DHSM 基础块"""
    
    def __init__(self, config: DHSMConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        self.state_layer = LinearStateLayer(config.hidden_dim, config.state_dim, config.dropout)
        
        self.attn = nn.MultiheadAttention(config.hidden_dim, config.num_heads, 
                                          dropout=config.dropout, batch_first=True)
        
        self.memory = HierarchicalMemory(config.hidden_dim, config.state_dim, 
                                         config.memory_levels, config.state_compression)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None,
                memory: Optional[List[torch.Tensor]] = None, use_attn: bool = True):
        
        x_state, new_state = self.state_layer(x, state)
        x = self.norm1(x + x_state)
        
        if use_attn:
            attn_out, _ = self.attn(x, x, x)
            x = self.norm2(x + attn_out)
        else:
            x = self.norm2(x)
        
        x_mem, new_memory = self.memory(x, memory)
        x = x + x_mem
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x, new_state, new_memory


class DHSM(nn.Module):
    """DHSM 完整模型"""
    
    def __init__(self, config: DHSMConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            DHSMBlock(config, i) for i in range(config.num_layers)
        ])
        
        if config.dynamic_depth:
            self.depth_router = DynamicDepthRouter(config.hidden_dim, config.num_layers, config.min_depth)
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, states: Optional[List[torch.Tensor]] = None,
                memories: Optional[List[List[torch.Tensor]]] = None):
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.emb_dropout(x)
        
        if states is None:
            states = [None] * self.config.num_layers
        if memories is None:
            memories = [None] * self.config.num_layers
        
        compute_depth = self.config.num_layers
        if hasattr(self, 'depth_router') and self.config.dynamic_depth:
            _, compute_depth = self.depth_router(x, self.training)
        
        new_states = []
        new_memories = []
        
        for i, block in enumerate(self.blocks):
            if i >= compute_depth:
                new_states.append(states[i] if states[i] is not None else None)
                new_memories.append(memories[i])
                continue
            
            x, state, memory = block(x, states[i], memories[i], use_attn=(i % 2 == 1))
            new_states.append(state)
            new_memories.append(memory)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_states, new_memories
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            'total': total,
            'total_M': total / 1e6,
            'total_B': total / 1e9
        }


if __name__ == '__main__':
    config = DHSMConfig(
        vocab_size=50257,
        hidden_dim=512,
        num_layers=6,
        state_dim=128,
        num_heads=8,
        dynamic_depth=False
    )
    
    model = DHSM(config)
    params = model.count_parameters()
    
    print(f'DHSM Model: {params["total_M"]:.2f}M parameters')
    
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, states, memories = model(x)
    
    print(f'Input: {x.shape} -> Output: {logits.shape}')
    print('DHSM core OK!')
