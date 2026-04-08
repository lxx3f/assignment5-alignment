# CS336 Assignment 5: 大语言模型对齐与推理强化学习

## 项目概述

本项目完成了 CS336 2025 年春季课程作业 5，实现了三种大语言模型对齐算法（SFT、Expert Iteration、GRPO），并在 MetaMathQA 数学推理数据集上进行了系统性的实验对比。通过使用 Qwen2.5-Math-1.5B 模型和 r1_zero 格式，最终实现了 **84.8%** 的数学推理准确率。

由于训练成本的限制，SFT、EI、GRPO训练的使用的样本量都不大，也没有调整超参数做对比，实际性能指标可能可以刷到更高。

## 实验结果总览

| 方法 | 正确率 | 格式错误率 | 训练数据量 | 训练时间 |
|------|--------|-----------|-----------|---------|
| **Baseline** | 23.0% | 54.6% | - | - |
| **SFT (20k)** | 82.4% | 0.8% | 20,000 | ~1小时 |
| **EI (3 iter)** | 84.2% | 0.0% | ~2,400/iter | ~2-3小时 |
| **GRPO** | 84.8% | 0.2% | 1,000 | ~1.5小时 |

## 核心实现

### 1. 监督微调 (SFT)
- 基于 Alpaca 格式的指令微调
- 使用 r1_zero prompt 模板强制输出格式：`<think>...</think> <answer>\boxed{...}</answer>`
- 20k 样本训练即可达到 82.4% 正确率，格式遵循率达 99.2%

### 2. 专家迭代 (Expert Iteration)
- 迭代式生成-验证-训练流程
- 每轮采样 1000 个问题，筛选正确答案作为下一轮训练数据
- 三轮迭代后正确率提升至 84.2%，格式错误率降为 0%

### 3. 组相对策略优化 (GRPO)
- 实现了 GRPO-Clip 损失函数，结合 PPO 的裁剪机制
- 组内归一化优势估计（无需 Critic 网络）
- 基于 EI 模型继续训练，达到 84.8% 正确率

### 4. 关键 Bug 修复
在 GRPO 训练过程中发现并修复了两个关键问题：
1. **is_correct 未定义错误**：奖励函数代码路径覆盖不全
2. **格式不匹配问题**：Prompt 要求 `<answer>123</answer>`，但奖励函数只认 `<answer>\boxed{123}</answer>`

详见 [GRPO_DEBUG.md](GRPO_DEBUG.md)

## 技术细节

### 训练配置
- **模型**: Qwen2.5-Math-1.5B (1.5B 参数)
- **数据集**: MetaMathQA (395K 样本，GSM8K + MATH 增强)
- **硬件**: RTX 5090 / A100
- **优化器**: AdamW，梯度累积
- **监控**: SwanLab 训练可视化

### 奖励函数设计
```python
r1_zero_reward_fn(response, ground_truth) -> {
    "format_reward": 1.0 if format_correct else 0.0,
    "answer_reward": 1.0 if answer_correct else 0.0,
    "reward": 1.0 if both else 0.0
}
```

### 核心超参数
| 参数 | SFT | EI | GRPO |
|------|-----|-----|------|
| Learning Rate | 2e-5 | 2e-5 | 1e-6 |
| Batch Size | 2 | 1 | 1 |
| Gradient Accumulation | 8 | 16 | 8 |
| Epochs | 2 | 1/iter | 3 |
| Temperature | - | 0.7 | 0.7 |
| Group Size | - | - | 4 |

## 项目结构

```
assignment5-alignment/
├── cs336_alignment/           # 核心算法实现
│   ├── grpo_utils.py         # GRPO 损失计算
│   ├── sft_utils.py          # SFT 训练工具
│   └── drgrpo_grader.py      # 奖励函数与答案验证
├── scripts/                   # 训练与评估脚本
│   ├── train_sft_metamath.py
│   ├── train_ei_metamath.py
│   ├── train_grpo_metamath.py
│   └── evaluate_metamath.py
├── data/MetaMathQA/          # 数据集
├── models/                   # 训练好的模型
├── results/                  # 评估结果
└── tests/                    # 单元测试
```

## 关键结论

1. **SFT 是对齐的基础**：20k 样本的监督微调解决了大部分格式问题（54.6% → 0.8%），并大幅提升准确率（+59.4%）

2. **数据质量过滤有效**：Expert Iteration 通过筛选正确答案进行迭代训练，边际提升 1.8%，且实现零格式错误

3. **GRPO 稳定继承**：基于 EI 模型的 GRPO 训练稳定达到 84.8% 正确率，验证了强化学习在数学推理任务上的有效性

4. **格式对齐至关重要**：Prompt 模板与奖励函数的格式要求必须严格一致，否则会严重影响训练效果

## 运行命令

### 基线评估
```bash
python scripts/evaluate_metamath.py \
    --model-name models/Qwen2.5-Math-1.5B \
    --data-path data/MetaMathQA/test.jsonl \
    --output results/baseline_eval.json \
    --max-samples 500
```

### SFT 训练
```bash
python scripts/train_sft_metamath.py \
    --model-name models/Qwen2.5-Math-1.5B \
    --train-data data/MetaMathQA/train.jsonl \
    --output-dir models/sft_metamath_20k \
    --num-epochs 2 \
    --max-samples 20000 \
    --batch-size 2 \
    --gradient-accumulation-steps 8
```

### Expert Iteration 训练
```bash
python scripts/train_ei_metamath.py \
    --model-name models/sft_metamath_20k/final \
    --output-dir models/ei_metamath \
    --num-iterations 3 \
    --samples-per-iteration 1000
```

### GRPO 训练
```bash
python scripts/train_grpo_metamath.py \
    --model-name models/ei_metamath/final \
    --output-dir models/grpo_metamath \
    --group-size 4 \
    --num-epochs 3 \
    --max-samples 1000
```

