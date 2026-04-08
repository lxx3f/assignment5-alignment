"""
MetaMathQA 数据集评估脚本

使用 r1_zero prompt 评估 Qwen 2.5 Math 1.5B 在 MetaMathQA 上的 zero-shot 性能。
支持三种奖励类别的统计分析和示例收集。
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Any
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 导入 drgrpo_grader 中的函数
sys.path.insert(0, str(Path(__file__).parent.parent / "cs336_alignment"))
from drgrpo_grader import r1_zero_reward_fn, extract_answer, grade


def format_r1_zero_prompt(question: str) -> str:
    """使用 r1_zero prompt 模板格式化问题"""
    template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    return template.format(question=question)


def extract_answer_from_response(response: str) -> str:
    """
    从 MetaMathQA 的 response 字段中提取标准答案
    """
    boxed_answer = extract_answer(response)
    if boxed_answer:
        return boxed_answer

    import re
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(',', '')

    match = re.search(r'[Tt]he answer is:\s*(.+?)(?:\n|$)', response)
    if match:
        answer = match.group(1).strip()
        return answer

    return ""


def categorize_sample(reward_info: dict) -> str:
    """
    将样本分类到三个类别之一

    类别1: "correct" - format_reward=1, answer_reward=1 (reward=1)
    类别2: "wrong_answer" - format_reward=1, answer_reward=0
    类别3: "wrong_format" - format_reward=0, answer_reward=0
    """
    format_reward = reward_info.get('format_reward', 0)
    answer_reward = reward_info.get('answer_reward', 0)

    if format_reward > 0 and answer_reward > 0:
        return "correct"
    elif format_reward > 0 and answer_reward == 0:
        return "wrong_answer"
    else:
        return "wrong_format"


def load_metamath_data(data_path: str, max_samples: int = None) -> list[dict[str, Any]]:
    """加载 MetaMathQA 数据（JSONL 格式）"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            ground_truth = extract_answer_from_response(item['response'])
            data.append({
                'question': item['query'],
                'ground_truth': ground_truth,
                'original_question': item.get('original_question', ''),
                'type': item.get('type', ''),
            })
    return data


def analyze_format_issue(response: str) -> dict:
    """
    分析格式问题的具体原因

    检查：
    1. 是否有 <think> 标签
    2. 是否有 </think> <answer> 标签
    3. 是否有 </answer> 标签
    4. 是否有 \boxed{} 在 <answer> 内
    """
    analysis = {
        'has_think_start': '<think>' in response,
        'has_think_end': '</think>' in response,
        'has_answer_start': '<answer>' in response,
        'has_answer_end': '</answer>' in response,
        'has_proper_format': '</think> <answer>' in response,
        'has_boxed': '\\boxed' in response,
        'response_length': len(response),
        'first_200_chars': response[:200],
    }
    return analysis


def evaluate_metamath(
    model_name: str,
    data_path: str,
    output_path: str,
    max_samples: int = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    device: str = "auto",
    num_examples_per_category: int = 10,
) -> dict[str, Any]:
    """
    评估模型在 MetaMathQA 数据集上的性能
    """
    print(f"\n{'='*70}")
    print(f"MetaMathQA Evaluation")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"Device: {device}")
    print(f"Examples per category: {num_examples_per_category}")
    print(f"{'='*70}\n")

    # 1. 加载模型和 tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    # 2. 加载数据
    test_data = load_metamath_data(data_path, max_samples=max_samples)
    print(f"Loaded {len(test_data)} test examples\n")

    # 3. 评估
    results = []
    category_counts = defaultdict(int)
    category_samples = defaultdict(list)

    print("Generating responses...")
    for idx, item in enumerate(tqdm(test_data)):
        question = item['question']
        ground_truth = item['ground_truth']

        # 格式化 prompt
        prompt = format_r1_zero_prompt(question)

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

        # 使用 r1_zero_reward_fn 评估
        reward_info = r1_zero_reward_fn(generated_text, ground_truth, fast=True)

        # 分类
        category = categorize_sample(reward_info)
        category_counts[category] += 1

        # 保存完整样本信息
        sample = {
            'index': idx,
            'question': question,
            'original_question': item['original_question'],
            'ground_truth': ground_truth,
            'type': item['type'],
            'generated_response': generated_text,
            'format_reward': reward_info['format_reward'],
            'answer_reward': reward_info['answer_reward'],
            'total_reward': reward_info['reward'],
            'category': category,
        }

        # 对格式错误的样本，额外分析原因
        if category == "wrong_format":
            sample['format_analysis'] = analyze_format_issue(generated_text)

        results.append(sample)
        category_samples[category].append(sample)

    # 4. 计算统计指标
    total = len(results)
    category_stats = {}
    for cat in ["correct", "wrong_answer", "wrong_format"]:
        count = category_counts[cat]
        category_stats[cat] = {
            'count': count,
            'percentage': count / total * 100 if total > 0 else 0
        }

    # 5. 收集每种类别的示例（用于分析）
    examples = {}
    for cat in ["correct", "wrong_answer", "wrong_format"]:
        # 随机选择指定数量的示例
        samples = category_samples[cat]
        if len(samples) > num_examples_per_category:
            selected = random.sample(samples, num_examples_per_category)
        else:
            selected = samples
        examples[cat] = selected

    # 6. 汇总结果
    evaluation_result = {
        'metadata': {
            'model_name': model_name,
            'data_path': data_path,
            'num_samples': total,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
        },
        'category_statistics': category_stats,
        'examples_by_category': examples,
        'all_results': results,
    }

    # 7. 保存结果
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, indent=2, ensure_ascii=False)

    # 8. 打印分析报告
    print_analysis_report(category_stats, examples, output_path)

    return evaluation_result


def print_analysis_report(category_stats: dict, examples: dict, output_path: str):
    """打印分析报告"""
    print(f"\n{'='*70}")
    print(f"                    评估结果分析报告")
    print(f"{'='*70}\n")

    total = sum(s['count'] for s in category_stats.values())
    print(f"总样本数: {total}")
    print()

    # 三类统计
    print("【一、奖励值分布统计】")
    print("-" * 50)
    print(f"类别1 (format=1, answer=1, reward=1):  {category_stats['correct']['count']:4d} ({category_stats['correct']['percentage']:5.1f}%)")
    print(f"类别2 (format=1, answer=0, reward=0):  {category_stats['wrong_answer']['count']:4d} ({category_stats['wrong_answer']['percentage']:5.1f}%)")
    print(f"类别3 (format=0, answer=0, reward=0):  {category_stats['wrong_format']['count']:4d} ({category_stats['wrong_format']['percentage']:5.1f}%)")
    print()

    # 格式错误分析
    print("【二、格式错误样本分析 (类别3)】")
    print("-" * 50)
    print(f"共收集 {len(examples['wrong_format'])} 个示例")
    print()
    print("观察要点：")
    print("- 模型是否输出 <think> 标签？")
    print("- 模型是否输出 </think> <answer> 序列？")
    print("- 模型是否输出 </answer> 结束标签？")
    print("- 问题可能原因：")
    print("  a) 基础模型本身未针对此格式训练")
    print("  b) prompt 模板与模型训练时的格式不匹配")
    print("  c) 生成长度不足，未完成格式")
    print()

    for i, ex in enumerate(examples['wrong_format'][:3], 1):
        print(f"  示例{i}:")
        print(f"    问题: {ex['question'][:80]}...")
        if 'format_analysis' in ex:
            analysis = ex['format_analysis']
            print(f"    格式分析:")
            print(f"      - 有 <think>: {analysis['has_think_start']}")
            print(f"      - 有 </think>: {analysis['has_think_end']}")
            print(f"      - 有 <answer>: {analysis['has_answer_start']}")
            print(f"      - 有 </answer>: {analysis['has_answer_end']}")
            print(f"      - 有正确格式序列: {analysis['has_proper_format']}")
            print(f"      - 生成长度: {analysis['response_length']} chars")
        print(f"    生成内容前200字: {ex['generated_response'][:200]}...")
        print()

    # 答案错误分析
    print("【三、答案错误样本分析 (类别2)】")
    print("-" * 50)
    print(f"共收集 {len(examples['wrong_answer'])} 个示例")
    print()
    print("观察要点：")
    print("- 模型是否正确理解了问题？")
    print("- 推理过程（<think>内）是否有逻辑错误？")
    print("- 最终答案与 ground truth 的差异类型？")
    print("  a) 数值计算错误")
    print("  b) 符号/正负号错误")
    print("  c) 单位/量纲错误")
    print("  d) 完全错误的思路")
    print()

    for i, ex in enumerate(examples['wrong_answer'][:3], 1):
        print(f"  示例{i}:")
        print(f"    问题: {ex['question'][:80]}...")
        print(f"    Ground Truth: {ex['ground_truth']}")
        # 尝试提取模型预测的答案
        import re
        answer_match = re.search(r'<answer>(.*?)</answer>', ex['generated_response'], re.DOTALL)
        if answer_match:
            predicted = answer_match.group(1).strip()
            print(f"    模型预测: {predicted}")
        print()

    # 正确样本
    print("【四、正确样本示例 (类别1)】")
    print("-" * 50)
    print(f"共收集 {len(examples['correct'])} 个示例")
    print()

    print("【五、结论与建议】")
    print("-" * 50)
    wrong_format_pct = category_stats['wrong_format']['percentage']
    wrong_answer_pct = category_stats['wrong_answer']['percentage']

    if wrong_format_pct > 50:
        print(f"⚠️  格式错误率高达 {wrong_format_pct:.1f}%，说明：")
        print("   - 基础模型 ZERO-SHOT 不擅长遵循 <think>/<answer> 格式")
        print("   - 需要通过 SFT 或 RL 训练来强化格式遵循能力")
    elif wrong_format_pct > 20:
        print(f"⚠️  格式错误率 {wrong_format_pct:.1f}%，需要关注格式训练")
    else:
        print(f"✓ 格式遵循良好 ({100-wrong_format_pct:.1f}% 合规)")

    print()

    if wrong_answer_pct > 30:
        print(f"⚠️  答案错误率 {wrong_answer_pct:.1f}%，说明数学推理能力有待提升")
    else:
        print(f"✓ 数学推理表现尚可 ({category_stats['correct']['percentage']:.1f}% 正确)")

    print()
    print(f"详细结果已保存至: {output_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MetaMathQA with category analysis")
    parser.add_argument("--model-name", type=str, default="models/Qwen2.5-Math-1.5B",
                        help="Model name or path")
    parser.add_argument("--data-path", type=str,
                        default="data/MetaMathQA/valid.jsonl",
                        help="Path to MetaMathQA data (JSONL format)")
    parser.add_argument("--output", type=str,
                        default="results/mathbaseline/metamath_analysis.json",
                        help="Output file path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy decoding)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--num-examples", type=int, default=10,
                        help="Number of examples to save per category")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling examples")

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    evaluate_metamath(
        model_name=args.model_name,
        data_path=args.data_path,
        output_path=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
        num_examples_per_category=args.num_examples,
    )


if __name__ == "__main__":
    main()
