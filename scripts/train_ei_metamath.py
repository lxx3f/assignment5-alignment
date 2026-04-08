"""
MetaMathQA Expert Iteration (专家迭代) 训练脚本

迭代流程：
1. 用当前模型生成 r1_zero 格式回答
2. 用 drgrpo_grader 验证答案正确性
3. 将正确的 (prompt, response) 加入训练集
4. 用新训练集微调模型（保持 r1_zero 格式）
5. 重复 N 轮
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import swanlab as wandb

# 导入 drgrpo_grader
sys.path.insert(0, str(Path(__file__).parent.parent / "cs336_alignment"))
from drgrpo_grader import r1_zero_reward_fn, extract_answer


def format_r1_zero_prompt(question: str) -> str:
    """使用 r1_zero prompt 模板"""
    template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    return template.format(question=question)


def extract_answer_from_response(response: str) -> str:
    """从 MetaMathQA response 中提取答案"""
    boxed_answer = extract_answer(response)
    if boxed_answer:
        return boxed_answer

    import re
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(',', '')

    match = re.search(r'[Tt]he answer is:\s*(.+?)(?:\n|$)', response)
    if match:
        return match.group(1).strip()

    return ""


def load_metamath_data(data_path: str, max_samples: int = None) -> list:
    """加载 MetaMathQA 数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            data.append({
                'question': item['query'],
                'answer': extract_answer_from_response(item['response']),
                'type': item.get('type', ''),
            })
    return data


def extract_reasoning_for_training(response: str) -> str:
    """从模型生成的 response 中提取推理过程用于训练"""
    # 移除 <think> 开始标签（如果模型生成了）
    response = response.replace('<think>', '').strip()

    # 提取 </think> 之前的部分作为推理
    if '</think>' in response:
        reasoning = response.split('</think>')[0].strip()
        return reasoning

    # 如果没有 </think>，尝试提取 <answer> 之前的部分
    if '<answer>' in response:
        reasoning = response.split('<answer>')[0].strip()
        return reasoning

    return response.strip()


class EISFTDataset(torch.utils.data.Dataset):
    """专家迭代 SFT 数据集（r1_zero 格式）"""

    def __init__(self, data: list, tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']

        # 构建训练用的完整文本
        # Prompt: ...Assistant: <think>
        # Completion: reasoning </think> <answer> answer </answer>
        prompt = format_r1_zero_prompt(question)

        # 从保存的 response 中提取推理和答案
        model_response = item['response']
        reasoning = extract_reasoning_for_training(model_response)
        answer = item['answer']

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

        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]

        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class MetaMathExpertIterationTrainer:
    """MetaMathQA 专家迭代训练器"""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        checkpoint_dir: str = None,
        num_iterations: int = 3,
        samples_per_iteration: int = 500,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        learning_rate: float = 2e-5,
        num_epochs_per_iteration: int = 1,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.num_iterations = num_iterations
        self.samples_per_iteration = samples_per_iteration
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        os.makedirs(output_dir, exist_ok=True)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    def generate_response(self, question: str) -> str:
        """用当前模型生成 r1_zero 格式回答"""
        prompt = format_r1_zero_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated_text

    def generate_training_data(self, questions: list) -> tuple:
        """
        生成训练数据：用当前模型回答问题，保留格式和答案都正确的样本
        """
        training_data = []
        stats = {'correct': 0, 'wrong_answer': 0, 'wrong_format': 0}

        # 随机采样问题
        sampled_questions = random.sample(
            questions,
            min(self.samples_per_iteration, len(questions))
        )

        for item in tqdm(sampled_questions, desc="Generating responses"):
            question = item['question']
            ground_truth = item['answer']

            # 生成回答
            response = self.generate_response(question)

            # 使用 r1_zero_reward_fn 评估
            reward_info = r1_zero_reward_fn(response, ground_truth, fast=True)

            # 分类统计
            if reward_info['reward'] > 0:
                category = 'correct'
            elif reward_info['format_reward'] > 0:
                category = 'wrong_answer'
            else:
                category = 'wrong_format'

            stats[category] += 1

            # 只保留完全正确的样本用于训练
            if reward_info['reward'] > 0:
                training_data.append({
                    'question': question,
                    'response': response,
                    'answer': ground_truth,
                })

        total = len(sampled_questions)
        print(f"\nGeneration statistics:")
        print(f"  Correct (format+answer): {stats['correct']}/{total} ({stats['correct']/total*100:.1f}%)")
        print(f"  Wrong answer only:       {stats['wrong_answer']}/{total} ({stats['wrong_answer']/total*100:.1f}%)")
        print(f"  Wrong format:            {stats['wrong_format']}/{total} ({stats['wrong_format']/total*100:.1f}%)")
        print(f"Generated {len(training_data)} training examples")

        return training_data, stats

    def train_on_data(self, training_data: list, iteration: int):
        """在新训练数据上微调模型"""
        print(f"\nIteration {iteration + 1}: Training on {len(training_data)} examples")

        dataset = EISFTDataset(training_data, self.tokenizer, max_length=2048)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.num_epochs_per_iteration // self.gradient_accumulation_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        self.model.train()
        total_loss = 0

        for epoch in range(self.num_epochs_per_iteration):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch + 1}")
            accum_steps = 0

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item() * self.gradient_accumulation_steps
                accum_steps += 1

                if accum_steps % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                progress_bar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss

        return total_loss / self.num_epochs_per_iteration

    def run(self, train_questions: list, use_wandb: bool = True):
        """运行专家迭代"""
        if use_wandb:
            wandb.init(project="cs336-alignment", name="ei-metamath-r1zero")

        for iteration in range(self.num_iterations):
            print(f"\n{'='*70}")
            print(f"Expert Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'='*70}")

            # 1. 生成训练数据
            training_data, stats = self.generate_training_data(train_questions)

            if len(training_data) == 0:
                print("Warning: No correct samples generated, skipping training")
                continue

            # 2. 微调模型
            avg_loss = self.train_on_data(training_data, iteration)

            # 3. 保存每轮模型（可选，如果设置了checkpoint_dir则保存）
            if hasattr(self, 'checkpoint_dir') and self.checkpoint_dir:
                save_path = os.path.join(self.checkpoint_dir, f"iteration-{iteration + 1}")
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                print(f"Saved iteration model to {save_path}")

            # 4. 记录结果
            if use_wandb:
                wandb.log({
                    'iteration': iteration + 1,
                    'train_examples': len(training_data),
                    'generate_accuracy': stats['correct'] / sum(stats.values()),
                    'format_compliance': (stats['correct'] + stats['wrong_answer']) / sum(stats.values()),
                    'train_loss': avg_loss,
                })

            print(f"Iteration {iteration + 1} completed. Saved to {save_path}")

        # 保存最终模型
        final_path = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"\nExpert Iteration completed! Final model saved to {final_path}")

        if use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration Training for MetaMathQA")
    parser.add_argument("--model-name", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-data", type=str, default="data/MetaMathQA/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/ei_metamath")
    parser.add_argument("--checkpoint-dir", type=str, default="/root/autodl-tmp/ei_checkpoints",
                        help="Directory for saving intermediate models")
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--samples-per-iteration", type=int, default=500)
    parser.add_argument("--num-epochs-per-iteration", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # 加载数据
    print(f"Loading training questions from: {args.train_data}")
    train_questions = load_metamath_data(args.train_data, max_samples=args.max_samples)
    print(f"Loaded {len(train_questions)} questions")

    # 创建训练器
    trainer = MetaMathExpertIterationTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_iterations=args.num_iterations,
        samples_per_iteration=args.samples_per_iteration,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        num_epochs_per_iteration=args.num_epochs_per_iteration,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 运行
    trainer.run(train_questions, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
