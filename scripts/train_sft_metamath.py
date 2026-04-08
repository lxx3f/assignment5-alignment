"""
MetaMathQA SFT (Supervised Fine-Tuning) 训练脚本

使用 r1_zero prompt 格式对 Qwen2.5-Math-1.5B 进行监督微调。
训练数据格式与评估一致，使用 <think>/<answer> 标签。
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import swanlab as wandb


def format_r1_zero_prompt(question: str, include_think_start: bool = True) -> str:
    """
    使用 r1_zero prompt 模板格式化问题

    Args:
        question: 问题文本
        include_think_start: 是否在 prompt 中包含 <think>（训练时设为 True）
    """
    template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant:{think_start}"""
    think_start = " <think>" if include_think_start else ""
    return template.format(question=question, think_start=think_start)


def extract_answer_from_response(response: str) -> str:
    """从 MetaMathQA 的 response 中提取答案（用于构建训练 target）"""
    # 尝试 \boxed{} 格式
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).strip()

    # 尝试 #### 格式
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(',', '')

    # 尝试 "The answer is: X" 格式
    match = re.search(r'[Tt]he answer is:\s*(.+?)(?:\n|$)', response)
    if match:
        return match.group(1).strip()

    return response.strip()


def load_metamath_data(data_path: str, max_samples: int = None):
    """加载 MetaMathQA 格式数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            data.append({
                'question': item['query'],
                'response': item['response'],
                'answer': extract_answer_from_response(item['response']),
                'type': item.get('type', ''),
            })
    return data


class R1ZeroSFTDataset(torch.utils.data.Dataset):
    """r1_zero 格式 SFT 数据集"""

    def __init__(self, data: list, tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        response = item['response']

        # 提取推理过程和答案
        # 从 response 中移除最后的 "The answer is: X" 或 #### 行
        reasoning = self._extract_reasoning(response)
        answer = item['answer']

        # 构建完整训练文本
        # Prompt: ...Assistant: <think>
        # Completion: reasoning </think> <answer> answer </answer>
        prompt = format_r1_zero_prompt(question, include_think_start=True)
        completion = f"{reasoning}</think> <answer>{answer}</answer>"

        full_text = prompt + completion

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Labels: 只计算 completion 部分的 loss
        labels = input_ids.clone()

        # 找到 prompt 的长度，将 prompt 部分的 label 设为 -100
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]

        # 将 prompt 部分设为 -100（不计算 loss）
        labels[:prompt_length] = -100
        # padding 部分也设为 -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def _extract_reasoning(self, response: str) -> str:
        """从 response 中提取推理过程（移除答案部分）"""
        # 移除 #### 行及其后的内容
        if '####' in response:
            return response.split('####')[0].strip() + '\n'

        # 移除 "The answer is:" 行及其后的内容
        match = re.search(r'([\s\S]*?)(?:[Tt]he answer is:[^\n]*\n?)$', response)
        if match:
            return match.group(1).strip() + '\n'

        # 移除 \boxed{} 后的 "The answer is:" 行
        lines = response.split('\n')
        for i, line in enumerate(reversed(lines)):
            if 'the answer is' in line.lower():
                return '\n'.join(lines[:-i-1]).strip() + '\n'

        return response.strip() + '\n'


def train_sft(
    model_name: str = "models/Qwen2.5-Math-1.5B",
    train_data_path: str = "data/MetaMathQA/train.jsonl",
    output_dir: str = "models/sft_metamath",
    checkpoint_dir: str = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    logging_steps: int = 10,
    save_steps: int = 500,
    max_samples: int = None,
    use_wandb: bool = True,
):
    """
    运行 MetaMathQA SFT 训练
    """
    # 初始化 wandb
    if use_wandb:
        wandb.init(project="cs336-alignment", name="sft-metamath-r1zero")

    # 加载模型和 tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # 加载数据
    print(f"Loading training data from: {train_data_path}")
    raw_data = load_metamath_data(train_data_path, max_samples=max_samples)
    print(f"Loaded {len(raw_data)} training examples")

    # 统计类型分布
    type_counts = {}
    for item in raw_data:
        t = item['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print("Data type distribution:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")

    # 创建数据集和 dataloader
    dataset = R1ZeroSFTDataset(raw_data, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 设置优化器和学习率调度
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # 训练循环
    model.train()
    global_step = 0
    total_loss = 0

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            # 将数据移到 GPU
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / gradient_accumulation_steps

            # 反向传播
            loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps
            epoch_loss += loss.item() * gradient_accumulation_steps

            # 梯度累积更新
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 记录日志
                if global_step % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    lr = scheduler.get_last_lr()[0]

                    if use_wandb:
                        wandb.log({
                            'loss': avg_loss,
                            'learning_rate': lr,
                            'epoch': epoch + step / len(dataloader),
                        })

                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
                    total_loss = 0

                # 保存 checkpoint
                if global_step % save_steps == 0:
                    save_dir = checkpoint_dir if checkpoint_dir else output_dir
                    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"\nSaved checkpoint to {save_path}")

        # 每个 epoch 结束保存
        save_dir = checkpoint_dir if checkpoint_dir else output_dir
        epoch_save_path = os.path.join(save_dir, f"epoch-{epoch+1}")
        model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        if use_wandb:
            wandb.log({'epoch_loss': avg_epoch_loss, 'epoch': epoch + 1})

    # 保存最终模型
    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="SFT Training for MetaMathQA with r1_zero format")
    parser.add_argument("--model-name", type=str, default="models/Qwen2.5-Math-1.5B",
                        help="Model name or path")
    parser.add_argument("--train-data", type=str, default="data/MetaMathQA/train.jsonl",
                        help="Training data path (MetaMathQA format)")
    parser.add_argument("--output-dir", type=str, default="models/sft_metamath",
                        help="Output directory for final model")
    parser.add_argument("--checkpoint-dir", type=str, default="/root/autodl-tmp/sft_checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of training samples")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")

    args = parser.parse_args()

    train_sft(
        model_name=args.model_name,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_samples=args.max_samples,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
