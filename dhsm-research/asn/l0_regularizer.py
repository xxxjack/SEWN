"""
L0 正则化器 - 连接级别稀疏控制
用于自适应地修剪不必要的连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class L0Regularizer(nn.Module):
    """
    L0 正则化模块
    
    通过可学习的门控参数实现连接级别的稀疏性
    使用 hard concrete 分布实现可微的离散门控
    """
    
    def __init__(
        self,
        num_features: int,
        # Hard Concrete 参数
        temperature: float = 0.5,
        stretch_limits: Tuple[float, float] = (-0.1, 1.1),
        # 初始化
        init_mean: float = 0.5,  # 初始门控概率
        # 正则化权重
        l0_weight: float = 1.0,
    ):
        """
        Args:
            num_features: 需要稀疏化的特征数量
            temperature: Hard Concrete 温度
            stretch_limits: Hard Concrete 的拉伸范围
            init_mean: 初始门控概率 (0-1)
            l0_weight: L0 正则化权重
        """
        super().__init__()
        
        self.num_features = num_features
        self.temperature = temperature
        self.stretch_limits = stretch_limits
        self.l0_weight = l0_weight
        
        # 计算 hard concrete 的参数
        self.lower, self.upper = stretch_limits
        assert self.lower < 0.0 and self.upper > 1.0
        
        # 可学习的 logit 参数 (控制门控概率)
        # 初始化: q ~ init_mean
        init_logit = math.log(init_mean / (1 - init_mean))
        self.logits = nn.Parameter(torch.full((num_features,), init_logit))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 L0 门控
        
        Args:
            x: 输入张量 [..., num_features]
            
        Returns:
            gated_x: 门控后的张量
            mask: 门控掩码 [..., num_features]
        """
        # 获取门控
        mask = self.get_gate(x.shape[:-1])
        
        # 应用门控
        gated_x = x * mask
        
        return gated_x, mask
    
    def get_gate(self, batch_shape: tuple = (), hard: bool = False) -> torch.Tensor:
        """
        获取门控掩码
        
        Args:
            batch_shape: 批次形状 (不包含 num_features 维度)
            hard: 是否使用硬门控 (推断时)
            
        Returns:
            mask: 门控掩码 [..., num_features]
        """
        # 确保批次形状是元组
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        
        if self.training and not hard:
            # Hard Concrete 采样
            # 扩展 logits 到批次维度
            expanded_logits = self.logits.expand(*batch_shape, -1) if batch_shape else self.logits
            
            u = torch.rand_like(expanded_logits)
            u = torch.clamp(u, 1e-6, 1 - 1e-6)  # 避免极端值
            
            # 转换到 hard concrete 分布
            s = torch.sigmoid((u.log() - (1 - u).log() + expanded_logits) / self.temperature)
            
            # 拉伸到 [lower, upper]
            s_bar = s * (self.upper - self.lower) + self.lower
            
            # 裁剪到 [0, 1]
            mask = F.hardtanh(s_bar, min_val=0.0, max_val=1.0)
        else:
            # 推断时: 直接使用期望值
            q = torch.sigmoid(self.logits)
            mask = (q > 0.5).float()  # 硬门控
            if batch_shape:
                mask = mask.expand(*batch_shape, -1)
        
        return mask
    
    def get_expected_sparsity(self) -> torch.Tensor:
        """计算期望的稀疏度 (非零比例)"""
        q = torch.sigmoid(self.logits)
        # 考虑 hard concrete 的裁剪
        # E[mask > 0] = q * (upper / (upper - lower))
        sparsity = q * (self.upper / (self.upper - self.lower))
        return torch.clamp(sparsity, 0, 1)
    
    def l0_penalty(self) -> torch.Tensor:
        """计算 L0 惩罚项"""
        return self.get_expected_sparsity().sum()
    
    def get_pruned_indices(self, threshold: float = 0.5) -> torch.Tensor:
        """获取被修剪的索引"""
        q = torch.sigmoid(self.logits)
        return (q > threshold).nonzero(as_tuple=True)[0]
    
    def get_active_indices(self, threshold: float = 0.5) -> torch.Tensor:
        """获取活跃的索引"""
        q = torch.sigmoid(self.logits)
        return (q > threshold).nonzero(as_tuple=True)[0]


class L0Linear(nn.Module):
    """
    带 L0 正则化的线性层
    
    自动在权重矩阵上应用 L0 门控
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 0.5,
        init_mean: float = 0.5,
        l0_weight: float = 1.0,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # L0 门控 (每个输出特征)
        self.l0_reg = L0Regularizer(
            out_features,
            temperature=temperature,
            init_mean=init_mean,
            l0_weight=l0_weight
        )
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带 L0 门控的前向传播"""
        # 应用门控到权重行
        mask = self.l0_reg.get_gate(x.shape[:-1] if x.dim() > 2 else ())
        
        # 门控权重
        masked_weight = self.weight * mask.unsqueeze(1)
        
        # 标准线性变换
        output = F.linear(x, masked_weight, self.bias)
        
        return output


class StructuredL0Regularizer(nn.Module):
    """
    结构化 L0 正则化器
    
    支持层级稀疏化 (整行/整列/整个模块)
    """
    
    def __init__(
        self,
        shape: tuple,
        mode: str = "row",  # "row", "col", "element", "group"
        temperature: float = 0.5,
        init_mean: float = 0.5,
        group_size: Optional[int] = None,
    ):
        """
        Args:
            shape: 需要稀疏化的张量形状
            mode: 稀疏化模式
            group_size: 分组稀疏化的组大小
        """
        super().__init__()
        
        self.shape = shape
        self.mode = mode
        self.group_size = group_size
        
        # 确定需要学习的参数数量
        if mode == "element":
            num_params = math.prod(shape)
        elif mode == "row":
            num_params = shape[0]
        elif mode == "col":
            num_params = shape[1]
        elif mode == "group":
            num_params = math.prod(shape) // group_size
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 初始化 logit
        init_logit = math.log(init_mean / (1 - init_mean))
        self.logits = nn.Parameter(torch.full((num_params,), init_logit))
        
        self.temperature = temperature
        self.lower, self.upper = -0.1, 1.1
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用结构化门控"""
        # 获取门控
        gate_prob = torch.sigmoid(self.logits)
        
        # 根据模式扩展门控
        if self.mode == "row":
            mask = gate_prob.view(-1, 1)
        elif self.mode == "col":
            mask = gate_prob.view(1, -1)
        elif self.mode == "element":
            mask = gate_prob.view(self.shape)
        elif self.mode == "group":
            mask = gate_prob.view(-1, 1).repeat(1, self.group_size).view(self.shape)
        
        # 应用门控
        if self.training:
            # Hard Concrete 采样
            u = torch.rand_like(mask)
            s = torch.sigmoid((u.log() - (1 - u).log() + mask.log()) / self.temperature)
            s_bar = s * (self.upper - self.lower) + self.lower
            mask = F.hardtanh(s_bar, 0, 1)
        else:
            mask = (gate_prob > 0.5).float()
            if self.mode != "element":
                mask = mask.view(-1, 1) if self.mode == "row" else mask.view(1, -1)
        
        gated_x = x * mask
        return gated_x, mask


# 便捷函数
def create_l0_layer(
    in_features: int,
    out_features: int,
    temperature: float = 0.5,
    init_sparsity: float = 0.5
) -> L0Linear:
    """创建带 L0 正则化的线性层"""
    return L0Linear(
        in_features, out_features,
        temperature=temperature,
        init_mean=init_sparsity
    )
