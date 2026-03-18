"""
Gumbel-Softmax 选择器 - 模块级别结构选择
用于在训练时实现可微的离散结构搜索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class GumbelSoftmaxSelector(nn.Module):
    """
    Gumbel-Softmax 模块选择器
    
    使用 Gumbel-Softmax 技巧实现可微的离散选择
    优点: 端到端可微，训练效率高
    缺点: 训练与推断时行为可能不一致
    """
    
    def __init__(
        self,
        num_modules: int,
        temperature: float = 1.0,
        hard: bool = False,
        gumbel_noise: bool = True,
    ):
        """
        Args:
            num_modules: 可选模块数量
            temperature: Gumbel-Softmax 温度 (越低越接近离散)
            hard: 训练时是否使用硬采样 (straight-through)
            gumbel_noise: 是否添加 Gumbel 噪声
        """
        super().__init__()
        
        self.num_modules = num_modules
        self.temperature = temperature
        self.hard = hard
        self.gumbel_noise = gumbel_noise
        
        # 可学习的 logit 权重
        self.logits = nn.Parameter(torch.zeros(num_modules))
    
    def forward(self, batch_size: int = 1, force_one_hot: bool = False) -> torch.Tensor:
        """
        执行模块选择
        
        Args:
            batch_size: 批次大小
            force_one_hot: 是否强制返回 one-hot (推断时)
            
        Returns:
            选择权重 [batch_size, num_modules]
        """
        # 获取 logits
        logits = self.logits.unsqueeze(0).expand(batch_size, -1)
        
        if self.training or force_one_hot:
            # Gumbel-Softmax 采样
            if self.gumbel_noise and self.training:
                # 添加 Gumbel 噪声
                gumbels = -torch.empty_like(logits).exponential_().log()
                logits = logits + gumbels
            
            # Softmax 归一化
            weights = F.softmax(logits / self.temperature, dim=-1)
            
            # Straight-through trick: 训练时强制 one-hot
            if self.hard and self.training:
                weights_hard = F.one_hot(weights.argmax(dim=-1), self.num_modules).float()
                weights = weights_hard - weights.detach() + weights
        else:
            # 推断时直接取 argmax
            weights = F.one_hot(logits.argmax(dim=-1), self.num_modules).float()
        
        return weights
    
    def get_selected_module_index(self, batch_size: int = 1) -> List[int]:
        """获取选中的模块索引 (用于推断)"""
        weights = self.forward(batch_size, force_one_hot=True)
        return weights.argmax(dim=-1).tolist()
    
    def get_importance_scores(self) -> torch.Tensor:
        """获取模块重要性分数 (基于累积的 logit)"""
        return F.softmax(self.logits, dim=0)


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator
    用于 Gumbel-Softmax 的硬采样梯度估计
    """
    
    @staticmethod
    def forward(ctx, input):
        # 直接返回 argmax (one-hot)
        idx = input.argmax(dim=-1)
        return F.one_hot(idx, input.size(-1)).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # 假设梯度均匀分布
        return grad_output.clone()


class MultiHeadGumbelSelector(nn.Module):
    """
    多头 Gumbel 选择器
    
    支持同时选择多个模块 (Top-K 选择)
    """
    
    def __init__(
        self,
        num_modules: int,
        num_heads: int = 1,
        temperature: float = 1.0,
        hard: bool = True,
    ):
        super().__init__()
        
        self.num_modules = num_modules
        self.num_heads = num_heads
        self.temperature = temperature
        self.hard = hard
        
        # 每个头独立的 logit
        self.logits = nn.Parameter(
            torch.zeros(num_heads, num_modules)
        )
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        返回选择权重 [batch_size, num_heads, num_modules]
        """
        logits = self.logits.unsqueeze(0).expand(batch_size, -1, -1)
        
        if self.training:
            # Gumbel-Softmax
            gumbels = -torch.empty_like(logits).exponential_().log()
            logits = logits + gumbels
            
            weights = F.softmax(logits / self.temperature, dim=-1)
            
            if self.hard:
                # Top-K 硬选择
                topk_weights = []
                for b in range(batch_size):
                    batch_weights = []
                    for h in range(self.num_heads):
                        # 选择概率最高的模块
                        idx = weights[b, h].argmax()
                        one_hot = F.one_hot(idx, self.num_modules).float()
                        batch_weights.append(one_hot)
                    topk_weights.append(torch.stack(batch_weights, dim=0))
                weights = torch.stack(topk_weights, dim=0)
        else:
            # 推断: Top-K
            topk_indices = logits.topk(self.num_heads, dim=-1)[1]
            weights = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
        
        return weights
    
    def get_selection_matrix(self, batch_size: int = 1) -> torch.Tensor:
        """获取选择矩阵 (batch_size, num_modules)"""
        selection = self.forward(batch_size)  # (B, H, M)
        # 如果是多头，取平均或任选其一
        if self.num_heads == 1:
            return selection.squeeze(1)
        else:
            return selection.mean(dim=1)  # (B, M)


# 便捷函数
def create_module_selector(
    num_modules: int,
    temperature: float = 1.0,
    hard: bool = True
) -> GumbelSoftmaxSelector:
    """创建模块选择器"""
    return GumbelSoftmaxSelector(num_modules, temperature, hard)
