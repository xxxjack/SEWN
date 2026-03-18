"""
复杂度评估器 - Complexity Estimator
评估输入任务的复杂度，用于决定架构初始化策略
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ComplexityLevel(Enum):
    """复杂度级别"""
    SIMPLE = "simple"      # <512 tokens, <10k vocab
    MEDIUM = "medium"      # 512-2048 tokens, 10k-50k vocab
    COMPLEX = "complex"    # >2048 tokens, >50k vocab


@dataclass
class ComplexityScore:
    """复杂度评分结果"""
    level: ComplexityLevel
    score: float  # 0.0-1.0 归一化分数
    token_count: int
    vocab_size: int
    details: Dict[str, float]


class ComplexityEstimator(nn.Module):
    """
    任务复杂度评估器
    
    使用规则 + 学习的混合策略评估输入复杂度
    
    可调参数:
        token_weight: Token数量的权重 (默认 0.6)
        vocab_weight: 词汇量的权重 (默认 0.4)
        简单阈值: simple_token_threshold=512, simple_vocab_threshold=10000
        复杂阈值: complex_token_threshold=2048, complex_vocab_threshold=50000
    """
    
    def __init__(
        self,
        # Token 阈值
        simple_token_threshold: int = 512,
        complex_token_threshold: int = 2048,
        # 词汇量阈值
        simple_vocab_threshold: int = 10000,
        complex_vocab_threshold: int = 50000,
        # 权重 (可调)
        token_weight: float = 0.6,
        vocab_weight: float = 0.4,
    ):
        super().__init__()
        
        self.simple_token_threshold = simple_token_threshold
        self.complex_token_threshold = complex_token_threshold
        self.simple_vocab_threshold = simple_vocab_threshold
        self.complex_vocab_threshold = complex_vocab_threshold
        self.token_weight = token_weight
        self.vocab_weight = vocab_weight
        
        # 可学习的复杂度预测器 (可选)
        self.use_learned = False
        self.learned_estimator = None
    
    def enable_learned_estimator(self, hidden_dim: int = 256):
        """启用可学习的复杂度预测器"""
        self.use_learned = True
        self.learned_estimator = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def estimate_token_count(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> int:
        """估算有效 token 数量"""
        mask = input_ids != pad_token_id
        return mask.sum().item()
    
    def estimate_vocab_size(self, input_ids: torch.Tensor, pad_token_id: int = 0) -> int:
        """估算词汇量大小"""
        mask = input_ids != pad_token_id
        unique_tokens = input_ids[mask].unique()
        return unique_tokens.numel()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        pad_token_id: int = 0
    ) -> ComplexityScore:
        """
        评估输入复杂度
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            hidden_states: 可选的隐藏状态 (用于可学习预测器)
            pad_token_id: padding token ID
            
        Returns:
            ComplexityScore: 复杂度评分结果
        """
        # Token 计数
        token_count = self.estimate_token_count(input_ids, pad_token_id)
        
        # 词汇量估算
        vocab_size = self.estimate_vocab_size(input_ids, pad_token_id)
        
        # 归一化分数计算
        token_score = self._normalize_token_count(token_count)
        vocab_score = self._normalize_vocab_size(vocab_size)
        
        # 加权综合分数
        rule_based_score = (
            self.token_weight * token_score + 
            self.vocab_weight * vocab_score
        )
        
        # 如果启用可学习预测器
        if self.use_learned and hidden_states is not None:
            learned_score = self.learned_estimator(hidden_states.mean(dim=1))
            # 融合规则和学习分数
            final_score = 0.7 * rule_based_score + 0.3 * learned_score.item()
        else:
            final_score = rule_based_score
        
        # 确定复杂度级别
        level = self._determine_level(final_score)
        
        return ComplexityScore(
            level=level,
            score=final_score,
            token_count=token_count,
            vocab_size=vocab_size,
            details={
                "token_score": token_score,
                "vocab_score": vocab_score,
                "token_count": token_count,
                "vocab_size": vocab_size
            }
        )
    
    def _normalize_token_count(self, count: int) -> float:
        """归一化 token 计数到 [0, 1]"""
        if count <= self.simple_token_threshold:
            return count / self.simple_token_threshold * 0.33
        elif count <= self.complex_token_threshold:
            # 线性插值 0.33-0.67
            ratio = (count - self.simple_token_threshold) / \
                    (self.complex_token_threshold - self.simple_token_threshold)
            return 0.33 + ratio * 0.34
        else:
            # 超过复杂阈值，归一化到 0.67-1.0
            ratio = min(1.0, (count - self.complex_token_threshold) / self.complex_token_threshold)
            return 0.67 + ratio * 0.33
    
    def _normalize_vocab_size(self, size: int) -> float:
        """归一化词汇量到 [0, 1]"""
        if size <= self.simple_vocab_threshold:
            return size / self.simple_vocab_threshold * 0.33
        elif size <= self.complex_vocab_threshold:
            ratio = (size - self.simple_vocab_threshold) / \
                    (self.complex_vocab_threshold - self.simple_vocab_threshold)
            return 0.33 + ratio * 0.34
        else:
            ratio = min(1.0, (size - self.complex_vocab_threshold) / self.complex_vocab_threshold)
            return 0.67 + ratio * 0.33
    
    def _determine_level(self, score: float) -> ComplexityLevel:
        """根据分数确定复杂度级别"""
        if score < 0.4:
            return ComplexityLevel.SIMPLE
        elif score < 0.7:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.COMPLEX
    
    def get_recommended_config(self, level: ComplexityLevel) -> Dict:
        """根据复杂度级别返回推荐配置"""
        configs = {
            ComplexityLevel.SIMPLE: {
                "num_worlds": 2,
                "metacognition_enabled": False,
                "dynamic_routing_enabled": True,
                "hidden_dim": 256,
                "num_layers": 2,
                "init_strategy": "lightweight"
            },
            ComplexityLevel.MEDIUM: {
                "num_worlds": 4,
                "metacognition_enabled": True,
                "dynamic_routing_enabled": True,
                "hidden_dim": 512,
                "num_layers": 4,
                "init_strategy": "balanced"
            },
            ComplexityLevel.COMPLEX: {
                "num_worlds": 8,
                "metacognition_enabled": True,
                "dynamic_routing_enabled": True,
                "hidden_dim": 768,
                "num_layers": 6,
                "init_strategy": "full"
            }
        }
        return configs[level]


# 便捷函数
def estimate_complexity(
    input_ids: torch.Tensor,
    pad_token_id: int = 0
) -> ComplexityScore:
    """快速评估输入复杂度"""
    estimator = ComplexityEstimator()
    return estimator(input_ids, pad_token_id=pad_token_id)
