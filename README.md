# SEWN: Self-Evolving World Model

A novel neural architecture for language modeling that combines consciousness, metacognition, and dynamic routing mechanisms.

## Overview

SEWN (Self-Evolving World Model) is an experimental architecture that introduces:

- **Consciousness Layer**: Selective attention and working memory
- **Metacognition Layer**: Self-monitoring and learning rate adaptation
- **Dynamic Routing**: Multi-world state management with learned routing

## Architecture

```
Input → Embedding → [SEWN Layer × 4] → Output
                      ↓
              ┌───────────────────┐
              │ Consciousness     │ → Attention weights
              │ Metacognition     │ → Confidence score
              │ Dynamic Routing   │ → World selection
              └───────────────────┘
```

### Configuration

| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Worlds per Layer | 4 |
| Hidden Dimension | 512 |
| Vocabulary Size | 10,000 |
| Max Sequence Length | 256 |
| Total Parameters | 24,415,796 |

## Experiments

### Experiment 9: SEWN Prototype (v0.1)

**Status**: ✅ Completed

| Metric | Value |
|--------|-------|
| Best Epoch | 8 |
| Best Validation Loss | 0.0006 |
| Confidence Score | 0.925 |
| Target Achievement | 83% improvement |

**Training Configuration**:
- Dataset: WikiText-103 (1.15M samples)
- Batch Size: 16
- Learning Rate: 8e-05
- Optimizer: AdamW
- Scheduler: CosineAnnealing
- Epochs: 10

**Overfitting Analysis**:
- Epoch 8 → 9: Val Loss increased 83% (0.0006 → 0.0011)
- Metacognition triggered at Confidence > 0.92
- Optimal early stopping point: Epoch 8

### Experiment 10: Ablation Study B (Metacognition)

**Status**: 🔄 In Progress

**Purpose**: Validate metacognition layer's contribution to overfitting prevention

**Ablation Configuration**:
- Metacognition Layer: **Disabled**
- Learning Rate Schedule: Fixed (no adaptive adjustment)

**Expected Result**: If metacognition is effective, overfitting should occur around Epoch 7-8

**Current Progress**:
- Started: 2026-03-15 22:58
- Epoch 1 Val Loss: 0.0014
- Status: Running

## Ablation Study Plan

| Priority | Experiment | Component | Status |
|----------|------------|-----------|--------|
| 1 | B | Metacognition Layer | 🔄 Running |
| 2 | C | Dynamic Routing | ⏳ Planned |
| 3 | A | Consciousness Layer | ⏳ Planned |
| 4 | D | Baseline Comparison | ⏳ Planned |

## Model Files

Due to size constraints, model weights are not included in this repository.

**Model Metadata**: See `models/model_metadata.json` for complete configuration and training details.

## Code Structure

```
.
├── experiment_06_wikitext.py      # Initial WikiText experiments
├── experiment_07_wikitext103.py   # WikiText-103 baseline
├── experiment_08_multiworld.py    # Multi-world SSM prototype
├── models/
│   └── model_metadata.json        # Model configuration & metrics
└── RESEARCH_REPORT.md             # Detailed experiment report
```

## Requirements

```
torch >= 2.0
transformers
datasets
numpy
tqdm
```

## Dataset

**WikiText-103**
- Training samples: 1,151,432
- Vocabulary size: 10,000 (custom tokenizer)
- Average sequence length: 256 tokens

## Training

Experiments were conducted on AutoDL GPU servers with RTX 6000 Ada (96GB VRAM).

```bash
python experiment_09_sewn_prototype.py --epochs 10 --batch_size 16 --lr 8e-5
```

## Results Summary

| Experiment | Description | Val Loss | Status |
|------------|-------------|----------|--------|
| Exp 8 | Multi-world SSM baseline | - | ✅ Complete |
| Exp 9 | SEWN Prototype | 0.0006 | ✅ Complete |
| Exp 10 | Ablation B (Metacognition) | - | 🔄 Running |

## Citation

If you use this code or find it helpful, please cite:

```bibtex
@misc{sewn2026,
  title={SEWN: Self-Evolving World Model},
  author={SEWN Research Team},
  year={2026},
  url={https://github.com/xxxjack/SEWN}
}
```

## License

MIT License

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
