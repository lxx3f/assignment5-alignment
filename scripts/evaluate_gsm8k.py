"""
GSM8K 评估脚本

评估模型在 GSM8K 数学推理数据集上的性能。
支持 few-shot 和 zero-shot 评估。
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


def format_alpaca_prompt(instruction: str) -> str:
    """使用 Alpaca 模板格式化 prompt"""
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    return template.format(instruction=instruction)


def normalize_number(num_str: str) -> str:
    """
    规范化数字字符串，用于比较。
    例如："3." -> "3", "03" -> "3", "3.0" -> "3", "10,000" -> "10000"
    """
    try:
        # 去掉首尾空白和逗号分隔符
        num_str = num_str.strip().replace(',', '')
        # 尝试转换为 float 再转回 string，去除不必要的小数点
        num = float(num_str)
        # 如果是整数，返回整数形式
        if num == int(num):
            return str(int(num))
        # 否则返回字符串形式，去掉末尾的 .0
        return str(num)
    except (ValueError, TypeError):
        return num_str.strip()


def extract_answer(text: str) -> str:
    """
    从文本中提取答案（查找 #### 后面的数字或最后一个数字）

    改进：
    1. 更严格的数字匹配（避免匹配 "3." 这样的不完整数字）
    2. 规范化数字格式用于比较
    """
    # 首先尝试查找 #### 格式（匹配 #### 后面的数字）
    # 匹配：整数（如 18, -5, 10,000）、小数（如 3.14, 0.5）
    # [\d,]+ 匹配数字和逗号（处理千分位），(?:\.\d+)? 匹配可选的小数部分
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        return normalize_number(match.group(1))

    # 如果没有 ####，尝试查找句子中的数字
    # 优先匹配 "The answer is X" 或 "answer is X" 格式
    answer_match = re.search(r'(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if answer_match:
        return normalize_number(answer_match.group(1))

    # 尝试匹配 "X dollars" 或 "$X" 格式（用于 GSM8K）
    # 注意：\$ 匹配字面量 $，而不是正则的行尾
    dollar_match = re.search(r'\$?(-?[\d,]+(?:\.\d+)?)\s*(?:dollars?|usd)', text, re.IGNORECASE)
    if dollar_match:
        return normalize_number(dollar_match.group(1))

    # 最后 fallback：返回最后一个看起来像数字的（更严格的模式）
    # 匹配整数或小数（支持逗号分隔），但不要以点开头的，也不要以点结尾的
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return normalize_number(numbers[-1])

    return ""


def load_gsm8k_data(data_path: str, max_samples: int = None) -> list[dict[str, Any]]:
    """加载 GSM8K 数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            data.append({
                'question': item['question'],
                'answer': item['answer'],
                'ground_truth': extract_answer(item['answer']),
            })
    return data


def evaluate_gsm8k(
    model_name: str,
    test_data_path: str,
    output_path: str,
    max_samples: int = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "auto",
) -> dict[str, Any]:
    """
    评估模型在 GSM8K 上的性能

    Args:
        model_name: 模型名称或路径
        test_data_path: 测试数据路径
        output_path: 输出结果路径
        max_samples: 最大评估样本数
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度（0 表示贪婪解码）
        device: 设备

    Returns:
        评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"GSM8K Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Data: {test_data_path}")
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

    # 2. 加载测试数据
    test_data = load_gsm8k_data(test_data_path, max_samples=max_samples)
    print(f"Loaded {len(test_data)} test examples\n")

    # 3. 评估
    results = []
    correct = 0
    total = 0

    print("Generating responses...")
    for item in tqdm(test_data):
        question = item['question']
        ground_truth = item['ground_truth']

        # 格式化 prompt
        prompt = format_alpaca_prompt(question)

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

        # 提取预测答案（会自动规范化）
        predicted_answer = extract_answer(generated_text)

        # 规范化 ground_truth 用于比较
        normalized_ground_truth = normalize_number(ground_truth)

        # 验证答案
        is_correct = predicted_answer == normalized_ground_truth
        if is_correct:
            correct += 1
        total += 1

        # 保存结果
        results.append({
            'question': question,
            'ground_truth_answer': item['answer'],
            'ground_truth_number': normalized_ground_truth,
            'generated_response': generated_text,
            'predicted_number': predicted_answer,
            'correct': is_correct,
        })

    # 4. 计算指标
    accuracy = correct / total if total > 0 else 0.0

    # 5. 汇总结果
    evaluation_result = {
        'metadata': {
            'model_name': model_name,
            'test_data_path': test_data_path,
            'num_samples': total,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
        },
        'metrics': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        },
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
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")

    return evaluation_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K")
    parser.add_argument("--model-name", type=str, default="models/Qwen2.5-Math-1.5B",
                        help="Model name or path")
    parser.add_argument("--test-data", type=str, default="data/gsm8k/test.jsonl",
                        help="Path to GSM8K test data")
    parser.add_argument("--output", type=str, default="outputs/gsm8k_results.json",
                        help="Output file path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cpu, cuda, auto)")

    args = parser.parse_args()

    evaluate_gsm8k(
        model_name=args.model_name,
        test_data_path=args.test_data,
        output_path=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
