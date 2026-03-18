# ASN - Adaptive Structure Network

自适应结构网络框架，核心理念：**结构即参数 (Structure as Parameters)**

## 概述

ASN 根据输入复杂度动态调整网络结构，实现：
- 简单任务 → 轻量架构（快速、低开销）
- 复杂任务 → 完整架构（高表达能力）

## 模块结构

```
asn/
├── __init__.py              # 模块导出
├── complexity_estimator.py  # 复杂度评估
├── gumbel_selector.py       # 模块级选择器
├── l0_regularizer.py        # 连接级稀疏化
├── asn_module.py            # ASN核心模块
└── test_asn.py              # 测试套件
```

## 核心组件

### 1. 复杂度估计器 (ComplexityEstimator)

评估输入复杂度，返回三个级别：

| 级别 | Token数 | 词汇量 | 推荐配置 |
|------|---------|--------|----------|
| SIMPLE | <512 | <10k | 2世界, 无元认知, 256维 |
| MEDIUM | 512-2048 | 10k-50k | 4世界, 有元认知, 512维 |
| COMPLEX | >2048 | >50k | 8世界, 有元认知, 768维 |

```python
from asn import ComplexityEstimator

estimator = ComplexityEstimator()
score = estimator(input_ids)
print(f"复杂度: {score.level}")  # SIMPLE/MEDIUM/COMPLEX
```

### 2. Gumbel-Softmax 选择器

模块级别的可微离散选择：

```python
from asn import GumbelSoftmaxSelector

selector = GumbelSoftmaxSelector(num_modules=4, temperature=1.0)
weights = selector(batch_size=2)  # 训练时软选择
indices = selector.get_selected_module_index()  # 推断时硬选择
```

### 3. L0 正则化器

连接级别的稀疏控制：

```python
from asn import L0Regularizer

regularizer = L0Regularizer(num_features=256)
gated_output, mask = regularizer(output)
sparsity = regularizer.get_expected_sparsity()
```

### 4. ASN 模块

整合所有组件：

```python
from asn import ASNModule, ASNConfig

config = ASNConfig(
    hidden_dim=512,
    num_worlds=4,
    metacognition_enabled=True
)

asn = ASNModule(config)
output, state = asn(hidden_states, input_ids, return_state=True)

print(f"复杂度级别: {state.selected_level}")
print(f"稀疏度: {state.sparsity_ratio:.2%}")
```

## 设计决策

### 混合冷启动策略 (方案C)

1. **简单任务**: 轻量启动，无元认知层
2. **中等任务**: 平衡启动，启用元认知
3. **复杂任务**: 完整架构，动态路由

### 与 SEWN 的关系

| 组件 | SEWN | ASN |
|------|------|-----|
| 多世界 | ✅ | ✅ |
| 元认知层 | 固定 | 自适应 |
| 动态路由 | ✅ | ✅ + 稀疏化 |
| 结构搜索 | ❌ | ✅ (Gumbel + L0) |

## 测试

```bash
cd /root/.openclaw/workspace/dhsm-research
source ../dhsm-env/bin/activate
python asn/test_asn.py
```

预期结果：**5/5 测试通过**

## 待审阅项

1. **复杂度阈值**: 当前为 512/2048 tokens，是否合理？
2. **温度参数**: Gumbel temperature=1.0, L0 temperature=0.5
3. **稀疏度目标**: 当前约 50%，是否需要调整？
4. **世界数量**: 简单2/中等4/复杂8，是否合理？

## 下一步

- [ ] 小钰审阅 (15:00)
- [ ] 集成到 SEWN 训练代码
- [ ] Exp10-ASN 实验 (对比固定架构)
- [ ] Exp13 实验 (Multi-News 复杂任务)

---

*ASN Framework v0.1.0 | 海蓝 🌊 2026-03-18*
