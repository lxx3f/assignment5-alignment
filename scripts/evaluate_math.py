"""
MATH 数据集评估脚本

使用 r1_zero prompt 评估 Qwen 2.5 Math 1.5B 在 MATH 验证集上的 zero-shot 性能。
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def format_r1_zero_prompt(question: str) -> str:
    """使用 r1_zero prompt 模板格式化问题"""
    template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    return template.format(question=question)


def extract_math_answer(text: str) -> str:
    """
    从模型输出中提取 MATH 格式的答案
    优先匹配 <answer> </answer> 标签内的内容
    """
    # 1. 优先提取 <answer> 标签内的内容
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        # 如果是 LaTeX 格式如 $\frac{1}{2}$，去掉 $ 符号
        answer = re.sub(r'^\$+|\$+$', '', answer)
        return answer

    # 2. 如果没有 </think> 标签，可能是生成未完成，尝试提取最后的内容
    # 但 MATH 数据集要求严格，优先返回空或提示
    return ""


def normalize_answer(answer: str) -> str:
    """
    规范化答案用于比较
    - 去除 LaTeX 包裹符号 $...$
    - 去除空格
    - 统一大小写（如果答案是文本）
    """
    answer = answer.strip()
    # 去除首尾 $ 符号
    answer = re.sub(r'^\$+|\$+$', '', answer)
    # 去除空格
    answer = answer.replace(' ', '')
    # 统一小写（对于文本答案）
    answer = answer.lower()
    return answer


def check_answer_correct(predicted: str, ground_truth: str) -> bool:
    """
    检查答案是否正确
    MATH 数据集答案可能有多种等价形式，这里使用简单的字符串匹配
    """
    pred_normalized = normalize_answer(predicted)
    truth_normalized = normalize_answer(ground_truth)

    # 直接匹配
    if pred_normalized == truth_normalized:
        return True

    # 尝试数值匹配（如果是数字）
    try:
        pred_num = float(pred_normalized)
        truth_num = float(truth_normalized)
        return abs(pred_num - truth_num) < 1e-6
    except (ValueError, TypeError):
        pass

    return False


def load_math_data(data_path: str, max_samples: int = None) -> list[dict[str, Any]]:
    """加载 MATH 验证数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            data.append({
                'problem': item['problem'],
                'solution': item.get('solution', ''),
                'answer': item['answer'],
                'level': item.get('level', ''),
                'type': item.get('type', ''),
            })
    return data


def evaluate_math(
    model_name: str,
    data_path: str,
    output_path: str,
    max_samples: int = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    device: str = "auto",
) -> dict[str, Any]:
    """
    评估模型在 MATH 数据集上的性能

    Args:
        model_name: 模型名称或路径
        data_path: MATH 数据路径
        output_path: 输出结果路径
        max_samples: 最大评估样本数
        max_new_tokens: 最大生成 token 数（MATH 题目需要更多 tokens）
        temperature: 采样温度
        device: 设备

    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"MATH Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # 1. 加载模型和 tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    # 2. 加载数据
    test_data = load_math_data(data_path, max_samples=max_samples)
    print(f"Loaded {len(test_data)} test examples\n")

    # 3. 评估
    results = []
    correct = 0
    total = 0

    print("Generating responses...")
    for item in tqdm(test_data):
        problem = item['problem']
        ground_truth = item['answer']

        # 格式化 prompt
        prompt = format_r1_zero_prompt(problem)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if device != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            if temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        # 解码生成的文本
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 提取预测答案
        predicted_answer = extract_math_answer(generated_text)

        # 验证答案
        is_correct = check_answer_correct(predicted_answer, ground_truth)
        if is_correct:
            correct += 1
        total += 1

        # 保存结果
        results.append({
            'problem': problem,
            'ground_truth_answer': ground_truth,
            'solution': item['solution'],
            'level': item['level'],
            'type': item['type'],
            'generated_response': generated_text,
            'predicted_answer': predicted_answer,
            'correct': is_correct,
        })

    # 4. 计算指标
    accuracy = correct / total if total > 0 else 0.0

    # 按难度等级统计
    level_stats = {}
    for r in results:
        level = r['level'] or 'unknown'
        if level not in level_stats:
            level_stats[level] = {'correct': 0, 'total': 0}
        level_stats[level]['total'] += 1
        if r['correct']:
            level_stats[level]['correct'] += 1

    for level in level_stats:
        stats = level_stats[level]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

    # 5. 汇总结果
    evaluation_result = {
        'metadata': {
            'model_name': model_name,
            'data_path': data_path,
            'num_samples': total,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
        },
        'metrics': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        },
        'level_breakdown': level_stats,
        'results': results,
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
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nAccuracy by level:")
    for level, stats in sorted(level_stats.items()):
        print(f"  Level {level}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")

    return evaluation_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MATH dataset")
    parser.add_argument("--model-name", type=str, default="models/Qwen2.5-Math-1.5B",
                        help="Model name or path")
    parser.add_argument("--data-path", type=str, default="/data/a5-alignment/MATH/validation.jsonl",
                        help="Path to MATH validation data")
    parser.add_argument("--output", type=str, default="outputs/math_results.json",
                        help="Output file path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cpu, cuda, auto)")

    args = parser.parse_args()

    evaluate_math(
        model_name=args.model_name,
        data_path=args.data_path,
        output_path=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
