# SEWN - Self-Evolving World Network

🧠 **自进化世界网络** - 具备意识和元认知能力的状态空间语言模型

## 架构演进

| 代次 | 名称 | 核心创新 |
|------|------|----------|
| 1 | SWM | 状态即世界 |
| 2 | DWSN | 多世界并行 |
| 3 | CESN | 意识涌现 |
| **4** | **SEWN** | **自进化架构** |

## SEWN v0.1 架构

### 五层设计

Layer 5: 进化引擎
Layer 4: 元认知层 - 置信度预测网络
Layer 3: 多世界SSM - 4个并行状态空间 [64,128,256,512]
Layer 2: 世界路由器 - 基于置信度的动态路由
Layer 1: 意识层 - Multi-head Attention (8 heads)

## 实验进展

| 实验 | 名称 | PPL | 状态 |
|------|------|-----|------|
| Exp1 | Baseline SSM | 11.85 | ✅ |
| Exp2 | Multi-scale SSM | 5.89 | ✅ |
| Exp3 | Dynamic Routing | 3.57 | ✅ |
| Exp4 | Capacity Management | 2.45 | ✅ |
| Exp5 | Full Architecture | 1.88 | ✅ |
| Exp6 | WikiText-2 | 3.09 | ✅ |
| Exp7 | WikiText-103 | - | ✅ |
| **Exp8** | **Multi-World SSM** | **8.31** | ✅ |
| **Exp9** | **SEWN Prototype** | **训练中** | 🔄 |

### 实验9 (SEWN Prototype) 训练指标

- **数据集**: WikiText-103 (115万训练样本)
- **架构**: 意识层 + 元认知层 + 动态路由v2
- **Loss下降**: 9.07 → 0.03 (下降99%+)
- **置信度**: 0.49 → 0.64 (元认知有效学习)

## 核心创新

### 1. 意识层
- Multi-head Attention识别关键token
- 8个注意力头，提取重要信息

### 2. 元认知层
- 预测模型对当前预测的置信度
- 动态调整路由温度
- 学习不确定性估计

### 3. 动态路由v2
- 低置信度 → 高温度 → 更均匀的路由
- 高置信度 → 低温度 → 更集中的路由

### 4. 多世界SSM

4个并行状态空间：
- World 1: dim=64 (细粒度)
- World 2: dim=128
- World 3: dim=256
- World 4: dim=512 (粗粒度)

## 快速开始

### 环境要求



### 运行实验



## 项目结构



## 引用



## 许可证

MIT License
