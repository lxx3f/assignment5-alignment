import re
from typing import Any


def parse_mmlu_response(mmlu_example: dict[str, Any], model_output: str) -> str | None:
    """
    从模型输出中解析 MMLU 答案选项（A, B, C, D）。

    Args:
        mmlu_example: MMLU 样例字典，包含 subject, question, options, answer
        model_output: 模型的输出文本

    Returns:
        解析出的选项字母（'A', 'B', 'C', 'D'），无法解析时返回 None
    """
    # 在 model_output 中搜索选项字母
    pattern = r"The correct answer is ([A-D])"
    match = re.search(pattern, model_output)
    if match:
        return match.group(1)
    return None


def parse_gsm8k_response(model_output: str) -> str | None:
    """
    从模型输出中提取 GSM8K 数值答案。

    通过查找输出中的最后一个数字来解析答案。

    Args:
        model_output: 模型的输出文本

    Returns:
        解析出的数值字符串，无法解析时返回 None
    """
    # 查找所有数字（包括整数、小数、负数）
    # 模式：可选的负号，数字，可选的小数部分
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, model_output)

    if not matches:
        return None

    # 返回最后一个数字
    return matches[-1]


import json
import random
from typing import Iterator
from torch.utils.data import Dataset, DataLoader
import torch




class PackedSFTDataset(Dataset):
    """
    打包式 SFT 数据集，将多个样本拼接成固定长度的序列。
    """

    def __init__(
        self,
        tokenizer,
        dataset_path: str,
        seq_length: int,
        shuffle: bool = False,
    ):
        """
        Args:
            tokenizer: 分词器
            dataset_path: 数据集文件路径（JSONL 格式）
            seq_length: 每个序列的长度
            shuffle: 是否打乱文档顺序
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # 读取数据
        documents = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt, response = data['prompt'], data['response']
                data_str = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n{response}"""
                data_str += '<|end_of_text|>'
                documents.append(data_str)

        # 打乱文档顺序
        if shuffle:
            random.shuffle(documents)

        # 将所有文本转换为 tokens
        all_tokens = []
        for doc in documents:
            tokens = tokenizer.encode(doc)
            all_tokens.extend(tokens)

        # 打包成固定长度的序列
        self.examples = []
        for i in range(0, len(all_tokens), seq_length):
            chunk = all_tokens[i:i + seq_length + 1]  # 多取一个用于 labels
            if len(chunk) < seq_length + 1:
                # 最后一段长度不足则直接丢弃
                break

            input_ids = chunk[:-1]
            labels = chunk[1:]

            self.examples.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_packed_sft_dataset(tokenizer, dataset_path: str, seq_length: int, shuffle: bool = False):
    """
    创建打包式 SFT 数据集。

    Args:
        tokenizer: Transformers 分词器
        dataset_path: 数据集文件路径
        seq_length: 序列长度
        shuffle: 是否打乱文档

    Returns:
        PyTorch Dataset
    """
    return PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        seq_length=seq_length,
        shuffle=shuffle,
    )


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
):
    """
    遍历数据集的批次。

    Args:
        dataset: PyTorch Dataset
        batch_size: 批次大小
        shuffle: 是否打乱

    Returns:
        DataLoader（支持 len() 和迭代）
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
