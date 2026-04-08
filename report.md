
# math_baseline 
All MetaMathQA data are augmented from the training sets of GSM8K and MATH. 
用MetaMathQA数据集做零样本推理基线。

拆分数据集：
```bash
python scripts/split_metamath_data.py \
    --input data/MetaMathQA/MetaMathQA-395K.json \
    --output-dir data/MetaMathQA \
    --train-ratio 0.8 \
    --valid-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42
Loading data from: data/MetaMathQA/MetaMathQA-395K.json
Total samples: 395000
Train: 316000 (80.0%)
Valid: 39500 (10.0%)
Test:  39500 (10.0%)
Saved train.jsonl: 316000 samples
Saved valid.jsonl: 39500 samples
Saved test.jsonl: 39500 samples

All files saved to: data/MetaMathQA
```

执行评估脚本：
```bash
Model: models/Qwen2.5-Math-1.5B
Data: data/MetaMathQA/valid.jsonl
Max samples: 500
Device: auto
Examples per category: 10
======================================================================

Loading model and tokenizer...
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 447.87it/s]
Loaded 500 test examples

Generating responses...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [25:36<00:00,  3.07s/it]
总样本数: 500

【奖励值分布统计】
--------------------------------------------------
类别1 (format=1, answer=1, reward=1):   115 ( 23.0%)
类别2 (format=1, answer=0, reward=0):   112 ( 22.4%)
类别3 (format=0, answer=0, reward=0):   273 ( 54.6%)

详细结果已保存至: results/mathbaseline/metamath_analysis.json

```

格式错误率高达 54.6%，说明：
   - 基础模型 ZERO-SHOT 不擅长遵循 <think>/<answer> 格式
   - 需要通过 SFT 或 RL 训练来强化格式遵循能力

数学推理表现尚可 (23.0% 正确)。

详细分析报告见：results/mathbaseline/analysis_report.md

# math sft
先小样本试试：
```bash
python scripts/train_sft_metamath.py \
    --model-name models/Qwen2.5-Math-1.5B \
    --train-data data/MetaMathQA/train.jsonl \
    --output-dir models/sft_test500 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2e-5 \
    --max-samples 500 \
    --max-length 1024 \
    --no-wandb
```
模型保存在：models/sft_test500
跑了五分钟
评估结果：
```bash
总样本数: 200

【一、奖励值分布统计】
--------------------------------------------------
类别1 (format=1, answer=1, reward=1):   170 ( 85.0%)
类别2 (format=1, answer=0, reward=0):    21 ( 10.5%)
类别3 (format=0, answer=0, reward=0):     9 (  4.5%)
```
具体见results/mathsft/sft_test500_eval.json
可以看到，格式错误率仅为 4.5%，说明：sft 模型在格式遵循能力上表现良好。而且只训练了500个样本就达到了不错的效果。

考虑训练成本，不跑全量样本，只跑 20000 个样本
```bash
python scripts/train_sft_metamath.py \
    --model-name models/Qwen2.5-Math-1.5B \
    --train-data data/MetaMathQA/train.jsonl \
    --output-dir models/sft_metamath_20k \
    --checkpoint-dir /root/autodl-tmp/sft_checkpoints \
    --num-epochs 2 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --max-samples 20000 \
    --max-length 1024 \
    --learning-rate 2e-5

Loading model: models/Qwen2.5-Math-1.5B
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 338/338 [00:00<00:00, 471.12it/s]
Loading training data from: data/MetaMathQA/train.jsonl
Loaded 20000 training examples
Data type distribution:
  GSM_AnsAug: 4099
  GSM_FOBAR: 2004
  GSM_Rephrased: 3994
  GSM_SV: 2045
  MATH_AnsAug: 3764
  MATH_FOBAR: 812
  MATH_Rephrased: 2539
  MATH_SV: 743

Starting training for 2 epochs...
Total steps: 2500, Warmup steps: 250
Epoch 1/2: 100%|████████████████████████████████████████████████████████████████████████████████| 10000/10000 [30:29<00:00,  5.46it/s, loss=1.2253, lr=1.11e-05]
Epoch 1 completed. Average loss: 0.1948
Epoch 2/2: 100%|████████████████████████████████████████████████████████████████████████████████| 10000/10000 [30:34<00:00,  5.45it/s, loss=1.2579, lr=0.00e+00]
Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.34s/it]

Epoch 2 completed. Average loss: 0.1619
Training completed! Final model saved to models/sft_metamath_20k/final
swanlab: 🏠 View project at https://swanlab.cn/@lx2323/cs336-alignment
swanlab: 🚀 View run at https://swanlab.cn/@lx2323/cs336-alignment/runs/30074icx4jnjlwsjv377n
```

评估结果：
```bash
python scripts/evaluate_metamath.py \
    --model-name models/sft_metamath_20k/final \
    --data-path data/MetaMathQA/test.jsonl \
    --output results/sft_20k_eval.json \
    --max-samples 500 \
    --max-new-tokens 2048 \
    --temperature 0.0

======================================================================
MetaMathQA Evaluation
======================================================================
Model: models/sft_metamath_20k/final
Data: data/MetaMathQA/test.jsonl
Max samples: 500
Device: auto
Examples per category: 10
======================================================================

总样本数: 500

【一、奖励值分布统计】
--------------------------------------------------
类别1 (format=1, answer=1, reward=1):   412 ( 82.4%)
类别2 (format=1, answer=0, reward=0):    84 ( 16.8%)
类别3 (format=0, answer=0, reward=0):     4 (  0.8%)

```
具体见results/mathsft/sft_test20k_eval.json


格式遵循良好 (99.2% 合规)
数学推理表现尚可 (82.4% 正确)

为什么正确率比上面还更低？

因为上面的评估只用了200个样本，而这次评估用了500个样本，不好比较

# math expert_iteration

```bash
python scripts/train_ei_metamath.py \
    --model-name models/sft_metamath_20k/final \
    --train-data data/MetaMathQA/train.jsonl \
    --output-dir models/ei_metamath \
    --checkpoint-dir /root/autodl-tmp/ei_checkpoints \
    --num-iterations 3 \
    --samples-per-iteration 1000 \
    --num-epochs-per-iteration 1 \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --max-new-tokens 512 \
    --temperature 0.7 \
    --learning-rate 2e-5

Loading training questions from: data/MetaMathQA/train.jsonl
Loaded 316000 questions
Loading model: models/sft_metamath_20k/final

swanlab: Tracking run with swanlab version 0.7.14
swanlab: Run data will be saved locally in /root/assignment5-alignment/swanlog/run-20260407_170641-jlppsgl58wk911kdv5y91
swanlab: 👋 Hi lx2323,welcome to swanlab!
swanlab: Syncing run ei-metamath-r1zero to the cloud
swanlab: 🏠 View project at https://swanlab.cn/@lx2323/cs336-alignment
swanlab: 🚀 View run at https://swanlab.cn/@lx2323/cs336-alignment/runs/jlppsgl58wk911kdv5y91

======================================================================
Expert Iteration 1/3
======================================================================

Generating responses: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [36:28<00:00,  2.19s/it]

Generation statistics:
  Correct (format+answer): 783/1000 (78.3%)
  Wrong answer only:       207/1000 (20.7%)
  Wrong format:            10/1000 (1.0%)
Generated 783 training examples

Iteration 1: Training on 783 examples
Training epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 783/783 [04:59<00:00,  2.62it/s, loss=0.0290]
Epoch 1 average loss: 0.0912
Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.38s/it]
Saved iteration model to /root/autodl-tmp/ei_checkpoints/iteration-1
Iteration 1 completed. Saved to /root/autodl-tmp/ei_checkpoints/iteration-1

======================================================================
Expert Iteration 2/3
======================================================================
Generating responses: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [34:42<00:00,  2.08s/it]

Generation statistics:
  Correct (format+answer): 835/1000 (83.5%)
  Wrong answer only:       156/1000 (15.6%)
  Wrong format:            9/1000 (0.9%)
Generated 835 training examples

Iteration 2: Training on 835 examples
Training epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 835/835 [02:52<00:00,  4.83it/s, loss=0.0773]
Epoch 1 average loss: 0.0584
======================================================================
Expert Iteration 3/3
======================================================================
Generating responses: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [36:27<00:00,  2.19s/it]

Generation statistics:
  Correct (format+answer): 819/1000 (81.9%)
  Wrong answer only:       164/1000 (16.4%)
  Wrong format:            17/1000 (1.7%)
Generated 819 training examples

Iteration 3: Training on 819 examples
Training epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 819/819 [02:49<00:00,  4.83it/s, loss=0.0478]
Epoch 1 average loss: 0.0424

Expert Iteration completed! Final model saved to models/ei_metamath/final
swanlab: 🏠 View project at https://swanlab.cn/@lx2323/cs336-alignment
swanlab: 🚀 View run at https://swanlab.cn/@lx2323/cs336-alignment/runs/jlppsgl58wk911kdv5y91
```

由于成本限制，采样1000个样本，每次迭代跑一个epoch，三轮迭代训练总共需要大约两到三小时。

评估模型：
```bash
python scripts/evaluate_metamath.py \
    --model-name models/ei_metamath/final \
    --data-path data/MetaMathQA/test.jsonl \
    --output results/ei_metamath_eval.json \
    --max-samples 500 \
    --max-new-tokens 2048 \
    --temperature 0.0

======================================================================
MetaMathQA Evaluation
======================================================================
Model: models/ei_metamath/final
Data: data/MetaMathQA/test.jsonl
Max samples: 500
Device: auto
Examples per category: 10
======================================================================
Generating responses...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [17:46<00:00,  2.13s/it]

总样本数: 500

【奖励值分布统计】
--------------------------------------------------
类别1 (format=1, answer=1, reward=1):   421 ( 84.2%)
类别2 (format=1, answer=0, reward=0):    79 ( 15.8%)
类别3 (format=0, answer=0, reward=0):     0 (  0.0%)
```

格式遵循良好 (100.0% 合规)

数学推理表现尚可 (84.2% 正确)


和前面的baseline和sft对比：
```bash
=== Baseline ===
Correct: 23.0% | Wrong Answer: 22.4% | Wrong Format: 54.6%
=== SFT 20k ===
Correct: 82.4% | Wrong Answer: 16.8% | Wrong Format: 0.8%
=== EI 3iter ===
Correct: 84.2% | Wrong Answer: 15.8% | Wrong Format: 0.0%
```
简单结论：
- SFT 解决格式问题：格式错误从 54.6% → 0.8%，r1_zero 格式训练非常有效
- SFT 大幅提升准确率：+59.4%，监督微调是核心提升来源
- EI 边际提升：+1.8%，但格式错误降为 0，数据质量过滤有一定作用


# GRPO

![[GRPO_DEBUG.md]]

```bash
python scripts/train_grpo_metamath.py \
    --model-name models/ei_metamath/final \
    --train-data data/MetaMathQA/train.jsonl \
    --output-dir models/grpo_metamath \
    --checkpoint-dir /root/autodl-tmp/grpo_checkpoints \
    --group-size 4 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --num-epochs 3 \
    --max-new-tokens 512 \
    --max-samples 200 \
    --temperature 0.7 \
    --learning-rate 1e-6

swanlab: Tracking run with swanlab version 0.7.14
swanlab: Run data will be saved locally in /root/assignment5-alignment/swanlog/run-20260407_213428-z4qx4rqw0fsal22yhhmca
swanlab: 👋 Hi lx2323,welcome to swanlab!
swanlab: Syncing run grpo-metamath to the cloud
swanlab: 🏠 View project at https://swanlab.cn/@lx2323/cs336-alignment
swanlab: 🚀 View run at https://swanlab.cn/@lx2323/cs336-alignment/runs/z4qx4rqw0fsal22yhhmca

Epoch 1/3
Epoch 1: 100%|█████████████████████████████████████████████████████████████████████| 200/200 [27:58<00:00,  8.39s/it, reward=1.000, format=100.00%, ans=100.00%]
Epoch 1 平均指标: {'loss': -0.029553135260939598, 'mean_reward': 0.78875, 'mean_advantage': 0.0, 'format_accuracy': 0.9925, 'answer_accuracy': 0.78875}

Epoch 2/3
Epoch 2: 100%|█████████████████████████████████████████████████████████████████████| 200/200 [28:43<00:00,  8.62s/it, reward=1.000, format=100.00%, ans=100.00%]
Epoch 2 平均指标: {'loss': -0.10696069408208132, 'mean_reward': 0.7775, 'mean_advantage': 0.0, 'format_accuracy': 0.99125, 'answer_accuracy': 0.7775}

Epoch 3/3
Epoch 3: 100%|█████████████████████████████████████████████████████████████████████| 200/200 [28:52<00:00,  8.66s/it, reward=1.000, format=100.00%, ans=100.00%]
Epoch 3 平均指标: {'loss': -0.1634947992861271, 'mean_reward': 0.77875, 'mean_advantage': 0.0, 'format_accuracy': 0.99125, 'answer_accuracy': 0.77875}
Final model saved to models/grpo_metamath/final
swanlab: 🏠 View project at https://swanlab.cn/@lx2323/cs336-alignment
swanlab: 🚀 View run at https://swanlab.cn/@lx2323/cs336-alignment/runs/z4qx4rqw0fsal22yhhmca
```

评估模型：
```bash
python scripts/evaluate_metamath.py \
    --model-name models/grpo_metamath/final \
    --data-path data/MetaMathQA/test.jsonl \
    --output results/grpo_metamath_eval.json \
    --max-samples 500 \
    --max-new-tokens 2048 \
    --temperature 0.0

======================================================================
MetaMathQA Evaluation
======================================================================
Model: models/grpo_metamath/final
Data: data/MetaMathQA/test.jsonl
Max samples: 500
Device: auto
Examples per category: 10
======================================================================
Generating responses...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [17:50<00:00,  2.14s/it]

总样本数: 500

【奖励值分布统计】
--------------------------------------------------
类别1 (format=1, answer=1, reward=1):   424 ( 84.8%)
类别2 (format=1, answer=0, reward=0):    75 ( 15.0%)
类别3 (format=0, answer=0, reward=0):     1 (  0.2%)

详细结果已保存至: results/grpo_metamath_eval.json
```

格式遵循良好 (99.8% 合规)
数学推理表现尚可 (84.8% 正确)