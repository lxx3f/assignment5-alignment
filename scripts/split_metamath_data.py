"""
MetaMathQA 数据集拆分脚本

将原始的 MetaMathQA-395K.json 拆分为 train/valid/test 三部分
默认比例：80% train, 10% valid, 10% test
"""

import argparse
import json
import os
import random
from pathlib import Path


def split_metamath_data(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    拆分 MetaMathQA 数据集

    Args:
        input_path: 原始 JSON 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    print(f"Loading data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"Total samples: {total}")

    # 设置随机种子
    random.seed(seed)
    indices = list(range(total))
    random.shuffle(indices)

    # 计算分割点
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_indices = set(indices[:train_end])
    valid_indices = set(indices[train_end:valid_end])
    test_indices = set(indices[valid_end:])

    # 拆分数据
    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in valid_indices]
    test_data = [data[i] for i in test_indices]

    print(f"Train: {len(train_data)} ({len(train_data)/total:.1%})")
    print(f"Valid: {len(valid_data)} ({len(valid_data)/total:.1%})")
    print(f"Test:  {len(test_data)} ({len(test_data)/total:.1%})")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 JSONL 格式（与 GSM8K 保持一致）
    splits = {
        'train.jsonl': train_data,
        'valid.jsonl': valid_data,
        'test.jsonl': test_data,
    }

    for filename, split_data in splits.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {filename}: {len(split_data)} samples")

    print(f"\nAll files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split MetaMathQA dataset")
    parser.add_argument("--input", type=str,
                        default="data/MetaMathQA/MetaMathQA-395K.json",
                        help="Input JSON file path")
    parser.add_argument("--output-dir", type=str,
                        default="data/MetaMathQA",
                        help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--valid-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    split_metamath_data(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
