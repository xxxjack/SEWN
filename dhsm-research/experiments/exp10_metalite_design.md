# Exp10-MetaLite 设计方案

## 实验目的

验证"元认知层 = 训练稳定器"假说

## 背景

### 实验10发现
- 无元认知层 → Val Loss 0.0001 (性能优秀)
- 但 Epoch 5 崩溃 (训练不稳定)
- **结论**: 元认知层防止训练崩溃，非性能提升

### 需要验证
1. 元认知层是否能防止崩溃？
2. 轻量元认知层是否足够？
3. 性能 vs 稳定性权衡？

## 实验设计

### 4组对比

| 组别 | 元认知层 | 学习率调节 | 目的 |
|------|----------|-----------|------|
| A | ❌ 无 | ❌ 无 | 基准 (复现实验10) |
| B | ✅ 完整 | ✅ 有 | 完整SEWN |
| C | ✅ 轻量 | ✅ 有 | MetaLite |
| D | ❌ 无 | ✅ 外部调度 | 对照组 |

### MetaLite 设计 (组C)

```python
class MetaLite(nn.Module):
    """轻量元认知层 - 只保留核心稳定功能"""
    def __init__(self, hidden_dim):
        super().__init__()
        # 简化：只做学习率调节
        self.lr_adjuster = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),  # 更小
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        self.base_lr = 1e-4
    
    def forward(self, hidden_states):
        # 计算学习率调整因子
        lr_factor = self.lr_adjuster(hidden_states.mean(dim=1))
        return self.base_lr * lr_factor
```

### 对比维度

| 指标 | 说明 |
|------|------|
| 最终 Val Loss | 性能 |
| 收敛 Epoch | 速度 |
| 训练稳定性 | 是否崩溃 |
| 参数量 | 效率 |
| 推理速度 | 部署友好度 |

## 数据集

### 主数据集: 合成 Multi-News
- 位置: `/root/.openclaw/workspace/dhsm-research/data/synthetic_multi_news/`
- 规模: train 1000 / val 200 / test 200

### 对照数据集: 简单任务
- 使用实验9/10的简单摘要任务
- 验证跨任务稳定性

## 超参数

```yaml
# 训练配置
batch_size: 8
max_epochs: 10
base_lr: 1e-4
warmup_steps: 100
gradient_clip: 1.0

# 早停
early_stopping_patience: 3
early_stopping_metric: val_loss

# 模型
hidden_dim: 256
num_layers: 2
num_worlds: 4
dropout: 0.1
```

## 预期结果

### 假设验证

| 假设 | 预期结果 |
|------|----------|
| 元认知层防崩溃 | B组训练稳定，A组可能崩溃 |
| MetaLite足够 | C组与B组稳定性相当 |
| 性能影响小 | B/C组性能接近A组 |
| 外部调度有效 | D组稳定性提升 |

### 成功标准

1. **主要**: C组(MetaLite)稳定性 = B组，参数量 < B组
2. **次要**: D组(外部调度)稳定性 > A组

## 执行计划

| 时间 | 任务 |
|------|------|
| 17:00-17:30 | 与小钰确认设计 |
| 17:30-18:00 | 代码实现 |
| 18:00-次日 | 训练运行 |

## 代码结构

```
experiments/
├── exp10_metalite_A.py  # 无元认知层
├── exp10_metalite_B.py  # 完整元认知层
├── exp10_metalite_C.py  # MetaLite
├── exp10_metalite_D.py  # 外部调度
└── exp10_metalite_compare.py  # 结果对比
```

---

*海蓝 🌊 | 2026-03-18 12:10 - MetaLite设计草稿*
