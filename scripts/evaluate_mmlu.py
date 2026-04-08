"""
MMLU 评估脚本
支持少量样本测试，结果可序列化保存
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    """格式化 MMLU 题目为模型输入 prompt"""
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(ord('A') + i)}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def parse_mmlu_answer(model_output: str) -> str | None:
    """
    从模型输出解析答案选项 (A, B, C, D)
    """
    import re

    patterns = [
        r'(?:answer|Answer)(?:\s+is)?[:\s]+([A-D])\b',
        r'(?:the\s+)?correct\s+(?:answer|option)(?:\s+is)?[:\s]+([A-D])\b',
        r'\(?([A-D])\)?[.:\s]',
        r'^[\s]*([A-D])[.:\s]',
    ]

    for pattern in patterns:
        match = re.search(pattern, model_output)
        if match:
            return match.group(1)

    # 直接查找单独的字母
    matches = re.findall(r'\b([A-D])\b', model_output)
    if matches:
        return matches[0]

    return None


def load_mmlu_data(data_dir: str, split: str = "test", max_samples: int | None = None,
                   subjects: list[str] | None = None) -> list[dict[str, Any]]:
    """
    加载 MMLU 数据

    Args:
        data_dir: MMLU 数据目录
        split: "dev", "val" 或 "test"
        max_samples: 最大样本数（None 表示全部）
        subjects: 指定学科列表（None 表示全部）
    """
    import csv

    data_dir = Path(data_dir)
    split_dir = data_dir / split

    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    examples = []

    # 获取所有 CSV 文件
    csv_files = list(split_dir.glob(f"*_{split}.csv"))

    if subjects:
        # 过滤指定学科
        csv_files = [f for f in csv_files if any(s in f.name for s in subjects)]

    print(f"Loading {len(csv_files)} subjects from {split} split...")

    for csv_file in sorted(csv_files):
        subject = csv_file.stem.replace(f"_{split}", "")

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 6:
                    continue

                examples.append({
                    "subject": subject,
                    "question": row[0],
                    "choices": row[1:5],
                    "answer": row[5],
                })

                if max_samples and len(examples) >= max_samples:
                    break

        if max_samples and len(examples) >= max_samples:
            break

    print(f"Loaded {len(examples)} examples")
    return examples


def evaluate_mmlu(
    model_path: str,
    data_dir: str,
    output_path: str,
    split: str = "test",
    max_samples: int | None = 10,
    subjects: list[str] | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 128,
    device: str = "auto",
) -> dict[str, Any]:
    """
    评估模型在 MMLU 上的性能

    Returns:
        dict 包含评估结果和元数据
    """

    print(f"\n{'='*60}")
    print(f"MMLU Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}/{split}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # 1. 加载模型和 tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 设置 pad_token（如果没有）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    # 2. 加载数据
    examples = load_mmlu_data(data_dir, split, max_samples, subjects)

    # 3. 生成和评估
    results = []
    correct = 0
    total = 0

    # 按学科统计
    subject_stats = {}

    print(f"\nGenerating responses...")
    start_time = time.time()

    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        # 格式化 prompt
        prompt = format_mmlu_prompt(example["question"], example["choices"])

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if device != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 使用贪心解码
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 解码输出
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 解析预测答案
        predicted = parse_mmlu_answer(model_output)
        ground_truth = example["answer"]
        is_correct = predicted == ground_truth

        if is_correct:
            correct += 1
        total += 1

        # 学科统计
        subject = example["subject"]
        if subject not in subject_stats:
            subject_stats[subject] = {"correct": 0, "total": 0}
        subject_stats[subject]["total"] += 1
        if is_correct:
            subject_stats[subject]["correct"] += 1

        # 保存结果
        result = {
            "index": i,
            "subject": subject,
            "question": example["question"],
            "choices": example["choices"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "model_output": model_output,
            "prompt": prompt,
        }
        results.append(result)

    elapsed_time = time.time() - start_time

    # 4. 计算指标
    overall_accuracy = correct / total if total > 0 else 0.0

    # 按学科计算准确率
    subject_accuracies = {}
    for subject, stats in subject_stats.items():
        subject_accuracies[subject] = stats["correct"] / stats["total"]

    # 5. 汇总结果
    evaluation_result = {
        "metadata": {
            "model_path": model_path,
            "data_dir": data_dir,
            "split": split,
            "max_samples": max_samples,
            "subjects": subjects,
            "num_samples": total,
            "elapsed_time_seconds": elapsed_time,
            "tokens_per_second": (total * max_new_tokens) / elapsed_time if elapsed_time > 0 else 0,
        },
        "metrics": {
            "overall_accuracy": overall_accuracy,
            "correct": correct,
            "total": total,
            "subject_accuracies": subject_accuracies,
        },
        "results": results,
    }

    # 6. 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Time: {elapsed_time:.2f}s ({elapsed_time/total:.2f}s per sample)")
    print(f"\nSubject Accuracies:")
    for subject, acc in sorted(subject_accuracies.items(), key=lambda x: -x[1]):
        count = subject_stats[subject]["total"]
        print(f"  {subject}: {acc:.2%} ({subject_stats[subject]['correct']}/{count})")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")

    return evaluation_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU")
    parser.add_argument("--model-path", type=str, default="models/Qwen2.5-Math-1.5B",
                        help="Path to the model")
    parser.add_argument("--data-dir", type=str, default="data/mmlu",
                        help="Path to MMLU data directory")
    parser.add_argument("--output", type=str, default="outputs/mmlu_results.json",
                        help="Output file path")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Maximum number of samples to evaluate (None for all)")
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Specific subjects to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for generation")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cpu, cuda, auto)")

    args = parser.parse_args()

    evaluate_mmlu(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_path=args.output,
        split=args.split,
        max_samples=args.max_samples,
        subjects=args.subjects,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
