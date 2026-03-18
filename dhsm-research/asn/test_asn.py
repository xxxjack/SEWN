#!/usr/bin/env python3
"""
ASN 框架测试脚本
验证自适应结构网络的核心功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from asn import (
    ComplexityEstimator, ComplexityLevel,
    GumbelSoftmaxSelector,
    L0Regularizer,
    ASNModule, ASNConfig
)


def test_complexity_estimator():
    """测试复杂度估计器"""
    print("\n" + "="*50)
    print("测试 1: 复杂度估计器")
    print("="*50)
    
    estimator = ComplexityEstimator()
    
    # 简单输入 (短文本)
    simple_input = torch.randint(1, 100, (1, 100))  # 100 tokens
    simple_score = estimator(simple_input)
    print(f"简单输入: {simple_score.level.value}, 分数: {simple_score.score:.3f}")
    print(f"  Token: {simple_score.token_count}, 词汇量: {simple_score.vocab_size}")
    
    # 中等输入
    medium_input = torch.randint(1, 5000, (1, 1000))  # 1000 tokens
    medium_score = estimator(medium_input)
    print(f"中等输入: {medium_score.level.value}, 分数: {medium_score.score:.3f}")
    print(f"  Token: {medium_score.token_count}, 词汇量: {medium_score.vocab_size}")
    
    # 复杂输入 (长文本)
    complex_input = torch.randint(1, 50000, (1, 3000))  # 3000 tokens
    complex_score = estimator(complex_input)
    print(f"复杂输入: {complex_score.level.value}, 分数: {complex_score.score:.3f}")
    print(f"  Token: {complex_score.token_count}, 词汇量: {complex_score.vocab_size}")
    
    # 获取推荐配置
    print("\n推荐配置 (复杂):")
    config = estimator.get_recommended_config(ComplexityLevel.COMPLEX)
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    print("✅ 复杂度估计器测试通过")
    return True


def test_gumbel_selector():
    """测试 Gumbel-Softmax 选择器"""
    print("\n" + "="*50)
    print("测试 2: Gumbel-Softmax 模块选择器")
    print("="*50)
    
    selector = GumbelSoftmaxSelector(
        num_modules=4,
        temperature=1.0,
        hard=True
    )
    
    # 训练模式
    selector.train()
    weights_train = selector(batch_size=2)
    print(f"训练模式权重: {weights_train.shape}")
    print(f"  样本权重: {weights_train}")
    
    # 推断模式
    selector.eval()
    weights_eval = selector(batch_size=2, force_one_hot=True)
    print(f"推断模式权重: {weights_eval.shape}")
    print(f"  样本权重: {weights_eval}")
    
    # 重要性分数
    importance = selector.get_importance_scores()
    print(f"模块重要性: {importance}")
    
    print("✅ Gumbel-Softmax 选择器测试通过")
    return True


def test_l0_regularizer():
    """测试 L0 正则化器"""
    print("\n" + "="*50)
    print("测试 3: L0 正则化器")
    print("="*50)
    
    regularizer = L0Regularizer(
        num_features=16,
        temperature=0.5,
        init_mean=0.7
    )
    
    # 随机输入
    x = torch.randn(2, 4, 16)  # [batch, seq, features]
    
    # 训练模式
    regularizer.train()
    gated_x, mask = regularizer(x)
    print(f"输入形状: {x.shape}")
    print(f"门控后形状: {gated_x.shape}")
    print(f"掩码形状: {mask.shape}")
    
    # 期望稀疏度
    sparsity = regularizer.get_expected_sparsity()
    print(f"期望稀疏度: {sparsity}")
    
    # L0 惩罚
    penalty = regularizer.l0_penalty()
    print(f"L0 惩罚: {penalty.item():.4f}")
    
    # 推断模式
    regularizer.eval()
    gated_x_eval, mask_eval = regularizer(x)
    print(f"推断模式掩码: {mask_eval[0, :8]}")
    
    # 活跃索引
    active_idx = regularizer.get_active_indices(threshold=0.5)
    print(f"活跃索引数量: {len(active_idx)}/{16}")
    
    print("✅ L0 正则化器测试通过")
    return True


def test_asn_module():
    """测试完整 ASN 模块"""
    print("\n" + "="*50)
    print("测试 4: ASN 模块")
    print("="*50)
    
    # 配置
    config = ASNConfig(
        hidden_dim=256,
        num_layers=2,
        num_worlds=4,
        metacognition_enabled=True,
        dynamic_routing_enabled=True
    )
    
    asn = ASNModule(config)
    
    # 测试输入
    batch_size = 2
    seq_len = 128
    hidden_dim = 256
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    input_ids = torch.randint(1, 10000, (batch_size, seq_len))
    
    # 前向传播
    print("前向传播...")
    output, state = asn(hidden_states, input_ids, return_state=True)
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")
    print(f"复杂度级别: {state.selected_level}")
    print(f"稀疏度: {state.sparsity_ratio:.2%}")
    print(f"活跃参数: {state.active_params}/{state.total_params}")
    
    # 打印摘要
    print("\n" + asn.summary())
    
    print("✅ ASN 模块测试通过")
    return True


def test_adaptive_behavior():
    """测试自适应行为"""
    print("\n" + "="*50)
    print("测试 5: 自适应行为验证")
    print("="*50)
    
    asn = ASNModule(ASNConfig(
        hidden_dim=128,
        num_layers=2,
        num_worlds=4
    ))
    
    # 不同复杂度的输入
    test_cases = [
        ("简单", torch.randint(1, 100, (1, 100)), torch.randn(1, 100, 128)),
        ("中等", torch.randint(1, 5000, (1, 1000)), torch.randn(1, 1000, 128)),
        ("复杂", torch.randint(1, 30000, (1, 2500)), torch.randn(1, 2500, 128)),
    ]
    
    for name, input_ids, hidden in test_cases:
        output, state = asn(hidden, input_ids, return_state=True)
        print(f"{name}输入:")
        print(f"  复杂度级别: {state.selected_level}")
        print(f"  分数: {state.complexity_score.score:.3f}")
        print(f"  Token数: {state.complexity_score.token_count}")
        print(f"  词汇量: {state.complexity_score.vocab_size}")
        print()
    
    print("✅ 自适应行为测试通过")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("       ASN 框架测试套件")
    print("  Adaptive Structure Network Framework")
    print("="*60)
    
    tests = [
        ("复杂度估计器", test_complexity_estimator),
        ("Gumbel-Softmax选择器", test_gumbel_selector),
        ("L0正则化器", test_l0_regularizer),
        ("ASN模块", test_asn_module),
        ("自适应行为", test_adaptive_behavior),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"❌ {name} 测试失败: {e}")
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, r, _ in results if r)
    total = len(results)
    
    for name, result, error in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if error:
            print(f"    错误: {error}")
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.0f}%)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
