"""
ASN 模块 - Adaptive Structure Network Core
整合复杂度估计、模块选择、L0正则化的完整ASN框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .complexity_estimator import ComplexityEstimator, ComplexityLevel, ComplexityScore
from .gumbel_selector import GumbelSoftmaxSelector
from .l0_regularizer import L0Regularizer


@dataclass
class ASNConfig:
    """ASN 配置"""
    # 基础配置
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    
    # 多世界配置
    num_worlds: int = 4
    world_dim: int = 128
    
    # 元认知配置
    metacognition_enabled: bool = True
    
    # 动态路由配置
    dynamic_routing_enabled: bool = True
    routing_iterations: int = 3
    
    # ASN 特有配置
    complexity_estimator_enabled: bool = True
    module_selector_enabled: bool = True
    l0_regularizer_enabled: bool = True
    
    # 初始化策略
    init_strategy: str = "balanced"  # "lightweight", "balanced", "full"
    
    # 温度参数
    gumbel_temperature: float = 1.0
    l0_temperature: float = 0.5


@dataclass
class ASNState:
    """ASN 运行状态"""
    complexity_score: Optional[ComplexityScore] = None
    selected_level: Optional[ComplexityLevel] = None
    module_selection_weights: Optional[torch.Tensor] = None
    l0_mask: Optional[torch.Tensor] = None
    active_params: int = 0
    total_params: int = 0
    sparsity_ratio: float = 0.0
    
    # 统计信息
    forward_time_ms: float = 0.0
    num_routing_iterations: int = 0


class ASNModule(nn.Module):
    """
    自适应结构网络模块
    
    根据输入复杂度动态调整网络结构
    """
    
    def __init__(self, config: ASNConfig):
        super().__init__()
        
        self.config = config
        
        # 1. 复杂度估计器
        if config.complexity_estimator_enabled:
            self.complexity_estimator = ComplexityEstimator()
        
        # 2. 模块选择器 (根据复杂度级别选择不同模块)
        if config.module_selector_enabled:
            # 3 个复杂度级别 × 多个可选模块
            self.module_selector = nn.ModuleDict({
                "simple": GumbelSoftmaxSelector(
                    num_modules=2,
                    temperature=config.gumbel_temperature
                ),
                "medium": GumbelSoftmaxSelector(
                    num_modules=4,
                    temperature=config.gumbel_temperature
                ),
                "complex": GumbelSoftmaxSelector(
                    num_modules=6,
                    temperature=config.gumbel_temperature
                ),
            })
        
        # 3. L0 正则化器 (连接级别稀疏)
        if config.l0_regularizer_enabled:
            self.l0_regularizer = L0Regularizer(
                num_features=config.hidden_dim,
                temperature=config.l0_temperature
            )
        
        # 4. 世界处理模块 (多世界架构)
        self.world_modules = self._build_world_modules()
        
        # 5. 动态路由
        if config.dynamic_routing_enabled:
            self.routing_gate = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.num_worlds),
                nn.Softmax(dim=-1)
            )
        
        # 6. 元认知层 (可选)
        if config.metacognition_enabled:
            self.metacognition = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.hidden_dim),
                nn.Sigmoid()
            )
        
        # 状态记录
        self.state = ASNState()
    
    def _build_world_modules(self) -> nn.ModuleDict:
        """构建多世界模块"""
        modules = nn.ModuleDict()
        
        # 简单配置
        modules["simple_core"] = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        
        # 中等配置
        modules["medium_core"] = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            )
            for _ in range(2)
        ])
        
        # 复杂配置
        modules["complex_core"] = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            )
            for _ in range(4)
        ])
        
        # 输出投影
        modules["output_proj"] = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        
        return modules
    
    def estimate_complexity(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None
    ) -> ComplexityScore:
        """估计输入复杂度"""
        if hasattr(self, 'complexity_estimator'):
            score = self.complexity_estimator(input_ids, hidden_states)
            self.state.complexity_score = score
            self.state.selected_level = score.level
            return score
        else:
            # 默认中等复杂度
            return ComplexityScore(
                level=ComplexityLevel.MEDIUM,
                score=0.5,
                token_count=1024,
                vocab_size=20000,
                details={}
            )
    
    def select_modules(self, batch_size: int) -> torch.Tensor:
        """根据复杂度选择模块"""
        if not hasattr(self, 'module_selector'):
            return None
        
        level = self.state.selected_level or ComplexityLevel.MEDIUM
        
        selector = self.module_selector[level.value]
        weights = selector(batch_size)
        
        self.state.module_selection_weights = weights
        return weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        return_state: bool = False
    ) -> Tuple[torch.Tensor, Optional[ASNState]]:
        """
        ASN 前向传播
        
        Args:
            hidden_states: 隐藏状态 [batch, seq, hidden]
            input_ids: 输入 token IDs (可选，用于复杂度估计)
            return_state: 是否返回状态
            
        Returns:
            output: 输出隐藏状态
            state: 运行状态 (可选)
        """
        batch_size = hidden_states.shape[0]
        
        # 1. 复杂度估计
        if input_ids is not None and hasattr(self, 'complexity_estimator'):
            self.estimate_complexity(input_ids, hidden_states.mean(dim=1, keepdim=True))
        
        # 2. 模块选择
        if hasattr(self, 'module_selector'):
            self.select_modules(batch_size)
        
        # 3. 世界处理
        level = self.state.selected_level or ComplexityLevel.MEDIUM
        
        if level == ComplexityLevel.SIMPLE:
            output = self.world_modules["simple_core"](hidden_states)
        elif level == ComplexityLevel.MEDIUM:
            output = hidden_states
            for module in self.world_modules["medium_core"]:
                output = module(output)
        else:  # COMPLEX
            output = hidden_states
            for module in self.world_modules["complex_core"]:
                output = module(output)
        
        # 4. L0 稀疏化
        if hasattr(self, 'l0_regularizer'):
            output_flat = output.view(-1, self.config.hidden_dim)
            output_flat, mask = self.l0_regularizer(output_flat)
            output = output_flat.view_as(hidden_states)
            self.state.l0_mask = mask
        else:
            # 默认 L0 掩码 (全1)
            self.state.l0_mask = torch.ones_like(hidden_states)
        
        # 5. 动态路由
        if self.config.dynamic_routing_enabled and hasattr(self, 'routing_gate'):
            # 简化: 使用平均池化后的路由
            pooled = output.mean(dim=1)  # [batch, hidden]
            routing_weights = self.routing_gate(
                torch.cat([pooled, pooled], dim=-1)
            )  # [batch, num_worlds]
            
            # 路由加权 - 对每个世界加权求和
            # output: [batch, seq, hidden], routing_weights: [batch, num_worlds]
            # 简化: 使用第一个世界权重
            output = output * routing_weights[:, 0].unsqueeze(1).unsqueeze(2)
            self.state.num_routing_iterations = 1
        else:
            routing_weights = None
        
        # 6. 元认知 (可选)
        if self.config.metacognition_enabled and hasattr(self, 'metacognition'):
            gating = self.metacognition(output.mean(dim=1, keepdim=True))
            output = output * gating
        
        # 7. 输出投影
        output = self.world_modules["output_proj"](output)
        
        # 更新统计
        self._update_stats()
        
        if return_state:
            return output, self.state
        return output
    
    def _update_stats(self):
        """更新统计信息"""
        if self.state.l0_mask is not None:
            active = (self.state.l0_mask > 0.5).sum().item()
            total = self.state.l0_mask.numel()
            self.state.active_params = active
            self.state.total_params = total
            self.state.sparsity_ratio = 1.0 - (active / total) if total > 0 else 0.0
    
    def get_config_for_level(self, level: ComplexityLevel) -> Dict[str, Any]:
        """获取指定复杂度级别的推荐配置"""
        base_config = {
            ComplexityLevel.SIMPLE: {
                "num_layers": 2,
                "hidden_dim": 256,
                "metacognition_enabled": False,
                "dropout": 0.1
            },
            ComplexityLevel.MEDIUM: {
                "num_layers": 4,
                "hidden_dim": 512,
                "metacognition_enabled": True,
                "dropout": 0.1
            },
            ComplexityLevel.COMPLEX: {
                "num_layers": 6,
                "hidden_dim": 768,
                "metacognition_enabled": True,
                "dropout": 0.15
            }
        }
        return base_config[level]
    
    def summary(self) -> str:
        """生成 ASN 状态摘要"""
        lines = [
            "=== ASN Module Summary ===",
            f"Config: {self.config}",
            f"Complexity Level: {self.state.selected_level}",
            f"Sparsity: {self.state.sparsity_ratio:.2%}",
            f"Active Params: {self.state.active_params}/{self.state.total_params}",
        ]
        if self.state.complexity_score:
            lines.append(f"Token Count: {self.state.complexity_score.token_count}")
            lines.append(f"Vocab Size: {self.state.complexity_score.vocab_size}")
        return "\n".join(lines)


class ASNWrapper(nn.Module):
    """
    ASN 包装器
    
    将 ASN 集成到现有模型中
    """
    
    def __init__(self, base_model: nn.Module, asn_config: ASNConfig):
        super().__init__()
        
        self.base_model = base_model
        self.asn = ASNModule(asn_config)
        
        # 获取基础模型输出维度
        self.hidden_dim = asn_config.hidden_dim
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """带 ASN 适配的前向传播"""
        # 基础模型编码
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        
        # ASN 处理
        asn_output, state = self.asn(
            hidden_states,
            input_ids=input_ids,
            return_state=True
        )
        
        return {
            "logits": asn_output,
            "asn_state": state,
            "base_outputs": outputs
        }


# 工厂函数
def create_asn_module(
    complexity: str = "balanced",
    hidden_dim: int = 512
) -> ASNModule:
    """创建 ASN 模块"""
    config = ASNConfig(
        hidden_dim=hidden_dim,
        init_strategy=complexity
    )
    return ASNModule(config)


def create_asn_from_estimate(
    complexity_score: ComplexityScore,
    hidden_dim: int = 512
) -> Tuple[ASNModule, Dict]:
    """从复杂度估计创建 ASN 模块"""
    level = complexity_score.level
    
    config = ASNConfig(
        hidden_dim=hidden_dim,
        init_strategy=level.value,
        metacognition_enabled=(level != ComplexityLevel.SIMPLE),
        num_worlds={
            ComplexityLevel.SIMPLE: 2,
            ComplexityLevel.MEDIUM: 4,
            ComplexityLevel.COMPLEX: 8
        }[level]
    )
    
    recommended_config = ComplexityEstimator().get_recommended_config(level)
    
    return ASNModule(config), recommended_config
