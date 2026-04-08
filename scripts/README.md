# 云端训练脚本使用指南

用于在 AutoDL 或其他 GPU 云平台上运行 CS336 Assignment 5 的训练。

## 环境准备

```bash
# 1. 创建并激活虚拟环境
conda create -n cs336 python=3.10
conda activate cs336

# 2. 安装依赖
pip install torch transformers accelerate wandb tqdm

# 3. 登录 wandb（可选）
wandb login
```

## 快速开始

### 1. GSM8K 零样本基线评估

```bash
python scripts/evaluate_gsm8k.py \
    --model-name Qwen/Qwen2.5-Math-1.5B \
    --test-data data/gsm8k/test.jsonl \
    --output outputs/gsm8k_baseline.json \
    --device cuda
```

### 2. SFT 训练

```bash
python scripts/train_sft.py \
    --model-name Qwen/Qwen2.5-Math-1.5B \
    --train-data data/gsm8k/train.jsonl \
    --output-dir outputs/sft \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5
```

### 3. Expert Iteration 训练

```bash
python scripts/train_expert_iteration.py \
    --model-name Qwen/Qwen2.5-Math-1.5B \
    --train-data data/gsm8k/train.jsonl \
    --output-dir outputs/expert_iteration \
    --num-iterations 3 \
    --samples-per-iteration 500 \
    --batch-size 4
```

### 4. GRPO 训练

```bash
python scripts/train_grpo.py \
    --model-name Qwen/Qwen2.5-Math-1.5B \
    --train-data data/gsm8k/train.jsonl \
    --output-dir outputs/grpo \
    --group-size 8 \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 1e-6
```

## 参数说明

### 评估脚本 (evaluate_gsm8k.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | Qwen/Qwen2.5-Math-1.5B | 模型名称或路径 |
| `--test-data` | data/gsm8k/test.jsonl | 测试数据路径 |
| `--output` | outputs/gsm8k_results.json | 输出结果路径 |
| `--max-samples` | None | 最大评估样本数（用于调试） |
| `--max-new-tokens` | 512 | 最大生成 token 数 |
| `--temperature` | 0.0 | 采样温度（0 表示贪婪解码） |
| `--device` | auto | 设备（cpu/cuda/auto） |

### SFT 训练 (train_sft.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | Qwen/Qwen2.5-Math-1.5B | 预训练模型名称 |
| `--train-data` | data/gsm8k/train.jsonl | 训练数据路径 |
| `--output-dir` | outputs/sft | 输出目录 |
| `--num-epochs` | 3 | 训练轮数 |
| `--batch-size` | 4 | 批次大小 |
| `--gradient-accumulation-steps` | 4 | 梯度累积步数 |
| `--learning-rate` | 2e-5 | 学习率 |
| `--max-length` | 2048 | 最大序列长度 |
| `--max-samples` | None | 最大训练样本数（用于调试） |
| `--no-wandb` | False | 禁用 wandb 日志 |

### Expert Iteration (train_expert_iteration.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-iterations` | 3 | 迭代轮数 |
| `--samples-per-iteration` | 500 | 每轮采样的问题数 |
| `--num-epochs-per-iteration` | 1 | 每轮训练的 epoch 数 |
| `--temperature` | 0.7 | 生成采样温度 |

### GRPO 训练 (train_grpo.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--group-size` | 8 | 每组 rollout 数量 |
| `--beta` | 0.1 | KL 惩罚系数 |
| `--learning-rate` | 1e-6 | 学习率 |

## 实验流程建议

### 阶段 1：基线评估
```bash
# 评估原始模型性能
python scripts/evaluate_gsm8k.py --output outputs/baseline.json
```

### 阶段 2：SFT 训练
```bash
# 使用 GSM8K 训练集进行 SFT
python scripts/train_sft.py --output-dir outputs/sft

# 评估 SFT 后的模型
python scripts/evaluate_gsm8k.py \
    --model-name outputs/sft/final \
    --output outputs/sft_results.json
```

### 阶段 3：Expert Iteration
```bash
# 运行专家迭代
python scripts/train_expert_iteration.py --output-dir outputs/ei

# 评估
python scripts/evaluate_gsm8k.py \
    --model-name outputs/ei/final \
    --output outputs/ei_results.json
```

### 阶段 4：GRPO
```bash
# 运行 GRPO 训练
python scripts/train_grpo.py --output-dir outputs/grpo

# 评估
python scripts/evaluate_gsm8k.py \
    --model-name outputs/grpo/final \
    --output outputs/grpo_results.json
```

## 结果对比

完成所有实验后，对比各方法的性能：

```python
import json

methods = ['baseline', 'sft', 'ei', 'grpo']
for method in methods:
    with open(f'outputs/{method}_results.json') as f:
        result = json.load(f)
    acc = result['metrics']['accuracy']
    print(f"{method.upper():12s}: {acc:.2%}")
```

## 注意事项

1. **显存需求**：1.5B 模型需要约 8-12GB 显存，批量大小可根据 GPU 调整
2. **训练时间**：
   - SFT：约 1-2 小时（3 epochs）
   - Expert Iteration：约 3-6 小时（3 iterations）
   - GRPO：约 6-12 小时（取决于 group size）
3. **调试建议**：先用 `--max-samples 100` 验证流程，再跑完整训练
