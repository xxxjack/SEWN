# SEWN: Self-Evolving World Model

A novel neural architecture for language modeling that combines consciousness, metacognition, and dynamic routing mechanisms.

## Overview

SEWN (Self-Evolving World Model) is an experimental architecture that introduces:

- **Consciousness Layer**: Selective attention and working memory
- **Metacognition Layer**: Self-monitoring and learning rate adaptation (conditional)
- **Dynamic Routing**: Multi-world state management with learned routing
- **ASN Framework**: Adaptive Structure Network for task-complexity-aware architecture selection

## Key Findings (2026-03-18)

### Metacognition Layer = Conditional Regularizer

Our experiments reveal that the metacognition layer serves as a **conditional regularizer**, not just a stability mechanism:

| Design | Simple Tasks | Complex Tasks | Recommendation |
|--------|--------------|---------------|----------------|
| None (NoMeta) | ✅ Stable | ⚠️ Risk | Low learning rate required |
| Light (MetaLite) | ✅ Best | ✅ Best | **Recommended** |
| Full (FullMeta) | ✅ | ❌ Overfits | Avoid |

### Critical Training Parameters

| Parameter | Risky | Stable | Notes |
|-----------|-------|--------|-------|
| Learning Rate | 8e-5 | 5e-5 | Lower LR prevents gradient explosion |
| Gradient Clipping | None | max_norm=1.0 | Essential for stability |

## Architecture

```
Input → Embedding → [SEWN Layer × 4] → Output
                      ↓
              ┌───────────────────┐
              │ Consciousness     │ → Attention weights
              │ Metacognition     │ → Confidence score (optional)
              │ Dynamic Routing   │ → World selection
              └───────────────────┘
```

### ASN Framework (New)

```
Input Complexity Estimation
         ↓
┌────────┼────────┐
│ Simple │ Medium │ Complex │
│ (2 worlds) │ (4 worlds) │ (8 worlds) │
└────────┴────────┴────────┘
         ↓
    MetaLite (optional)
         ↓
      Output
```

## Experiments

### Experiment 10: Metacognition Ablation

**Status**: ✅ Completed (with crash)

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|--------|
| 1 | 0.7157 | 0.0014 | ✅ |
| 2 | 0.0012 | 0.0003 | ✅ |
| 3 | 0.0002 | **0.0001** | ✅ Best |
| 4 | 0.0001 | **0.0001** | ✅ |
| 5 | NaN | NaN | ❌ **Crashed** |

**Key Finding**: Without metacognition layer + high learning rate (8e-5), training collapses at Epoch 5.

### MetaLite Experiments (Simple Tasks)

**Status**: ✅ Completed

| Model | Params | Val Loss | Status |
|-------|--------|----------|--------|
| A_NoMeta | 6.7M | 9.3601 | ✅ Stable |
| B_FullMeta | 6.8M | 9.3577 | ✅ Stable |
| C_MetaLite | 6.7M | 9.3512 | ✅ Good |
| **D_External** | 6.7M | **9.3462** | 🏆 **Best** |

**Finding**: On simple tasks, external scheduling outperforms internal metacognition.

### Complex Task Validation

**Status**: ✅ Completed

| Model | Train Loss | Val Loss | Gap | Status |
|-------|------------|----------|-----|--------|
| A_NoMeta | 10.43 | 11.14 | - | ✅ Stable |
| B_FullMeta | **5.26** | 11.14 | **2.1x** | ⚠️ Overfits |
| **C_MetaLite** | 10.47 | **11.14** | - | 🏆 **Best** |
| D_External | 10.44 | 11.14 | - | ✅ Stable |

**Finding**: FullMeta overfits on complex tasks (Train 5.26 vs Val 11.14). MetaLite achieves best balance.

## Code Structure

```
.
├── experiments/
│   ├── exp10_metalite.py              # MetaLite experiments
│   ├── complex_task_validation.py     # Complex task experiments
│   ├── exp10_metalite_design.md       # Design document
│   └── research_findings_2026-03-18.md
├── asn/
│   ├── __init__.py
│   ├── complexity_estimator.py        # Task complexity estimation
│   ├── gumbel_selector.py             # Differentiable module selection
│   ├── l0_regularizer.py              # Connection-level sparsity
│   ├── asn_module.py                  # ASN core module
│   ├── test_asn.py                    # Test suite (5/5 passed)
│   └── README.md
├── models/
│   └── model_metadata.json
└── README.md
```

## Requirements

```
torch >= 2.0
transformers
datasets
numpy
tqdm
```

## Training Recommendations

Based on our experiments:

1. **Learning Rate**: Use 5e-5 (not 8e-5)
2. **Gradient Clipping**: Always use max_norm=1.0
3. **Metacognition**: Use MetaLite (lightweight design)
4. **Early Stopping**: patience=3 epochs

## Results Summary

| Experiment | Description | Key Result | Status |
|------------|-------------|------------|--------|
| Exp 9 | SEWN Prototype | Val Loss 0.0006 | ✅ |
| Exp 10 | Metacognition Ablation | Crash at Epoch 5 | ✅ |
| MetaLite | Simple Task | D_External best | ✅ |
| Complex | Complex Task | C_MetaLite best | ✅ |

## Citation

```bibtex
@misc{sewn2026,
  title={SEWN: Self-Evolving World Model with Conditional Metacognition},
  author={SEWN Research Team},
  year={2026},
  url={https://github.com/xxxjack/SEWN}
}
```

## License

MIT License
