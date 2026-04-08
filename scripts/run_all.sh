#!/bin/bash
# 一键运行所有实验脚本
# 用于 AutoDL 云端 GPU 环境

set -e  # 遇到错误立即退出

echo "========================================"
echo "CS336 Assignment 5 - 云端训练脚本"
echo "========================================"

# 配置
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B}"
TRAIN_DATA="${TRAIN_DATA:-data/gsm8k/train.jsonl}"
TEST_DATA="${TEST_DATA:-data/gsm8k/test.jsonl}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs}"

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

echo ""
echo "配置信息:"
echo "  Model: $MODEL_NAME"
echo "  Train data: $TRAIN_DATA"
echo "  Test data: $TEST_DATA"
echo "  Output: $OUTPUT_BASE"
echo ""

# 阶段 1: 基线评估
echo "========================================"
echo "阶段 1: GSM8K 基线评估"
echo "========================================"
python scripts/evaluate_gsm8k.py \
    --model-name "$MODEL_NAME" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_BASE/baseline_results.json" \
    --device cuda

echo ""
echo "基线评估完成!"
cat "$OUTPUT_BASE/baseline_results.json" | grep -A 5 '"metrics"'
echo ""

# 阶段 2: SFT 训练
echo "========================================"
echo "阶段 2: SFT 训练"
echo "========================================"
python scripts/train_sft.py \
    --model-name "$MODEL_NAME" \
    --train-data "$TRAIN_DATA" \
    --output-dir "$OUTPUT_BASE/sft" \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5

echo ""
echo "SFT 训练完成!"
echo ""

# 评估 SFT 模型
echo "评估 SFT 模型..."
python scripts/evaluate_gsm8k.py \
    --model-name "$OUTPUT_BASE/sft/final" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_BASE/sft_results.json" \
    --device cuda

echo ""
echo "SFT 评估完成!"
cat "$OUTPUT_BASE/sft_results.json" | grep -A 5 '"metrics"'
echo ""

# 阶段 3: Expert Iteration
echo "========================================"
echo "阶段 3: Expert Iteration"
echo "========================================"
python scripts/train_expert_iteration.py \
    --model-name "$MODEL_NAME" \
    --train-data "$TRAIN_DATA" \
    --output-dir "$OUTPUT_BASE/expert_iteration" \
    --num-iterations 3 \
    --samples-per-iteration 500 \
    --batch-size 4

echo ""
echo "Expert Iteration 完成!"
echo ""

# 评估 EI 模型
echo "评估 Expert Iteration 模型..."
python scripts/evaluate_gsm8k.py \
    --model-name "$OUTPUT_BASE/expert_iteration/final" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_BASE/ei_results.json" \
    --device cuda

echo ""
echo "EI 评估完成!"
cat "$OUTPUT_BASE/ei_results.json" | grep -A 5 '"metrics"'
echo ""

# 阶段 4: GRPO
echo "========================================"
echo "阶段 4: GRPO 训练"
echo "========================================"
python scripts/train_grpo.py \
    --model-name "$MODEL_NAME" \
    --train-data "$TRAIN_DATA" \
    --output-dir "$OUTPUT_BASE/grpo" \
    --group-size 8 \
    --batch-size 4 \
    --num-epochs 3 \
    --learning-rate 1e-6

echo ""
echo "GRPO 训练完成!"
echo ""

# 评估 GRPO 模型
echo "评估 GRPO 模型..."
python scripts/evaluate_gsm8k.py \
    --model-name "$OUTPUT_BASE/grpo/final" \
    --test-data "$TEST_DATA" \
    --output "$OUTPUT_BASE/grpo_results.json" \
    --device cuda

echo ""
echo "GRPO 评估完成!"
cat "$OUTPUT_BASE/grpo_results.json" | grep -A 5 '"metrics"'
echo ""

# 最终结果汇总
echo "========================================"
echo "实验结果汇总"
echo "========================================"
echo ""
echo "方法                准确率"
echo "-------------------- --------"

for method in baseline sft ei grpo; do
    result_file="$OUTPUT_BASE/${method}_results.json"
    if [ -f "$result_file" ]; then
        accuracy=$(python -c "import json; print(f\"{json.load(open('$result_file'))['metrics']['accuracy']:.2%}\")")
        printf "%-20s %s\n" "$method" "$accuracy"
    fi
done

echo ""
echo "所有实验完成! 结果保存在 $OUTPUT_BASE/"
echo "========================================"
