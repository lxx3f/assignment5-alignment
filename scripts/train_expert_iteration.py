"""
Expert Iteration (专家迭代) 训练脚本

迭代流程：
1. 用当前模型生成回答
2. 用 verifier 验证答案正确性
3. 将正确的 (prompt, response) 加入训练集
4. 用新训练集微调模型
5. 重复 N 轮
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
import wandb


def format_alpaca_prompt(instruction: str, response: str = "") -> str:
    """使用 Alpaca 模板格式化 prompt"""
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""
    return template.format(instruction=instruction, response=response)


def normalize_number(num_str: str) -> str:
    """
    规范化数字字符串，用于比较。
    例如："3." -> "3", "03" -> "3", "3.0" -> "3", "10,000" -> "10000"
    """
    try:
        num_str = num_str.strip().replace(',', '')
        num = float(num_str)
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        return num_str.strip()


def extract_answer(text: str) -> str:
    """
    从文本中提取答案（查找 #### 后面的数字或最后一个数字）
    """
    # 首先尝试查找 #### 格式
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        return normalize_number(match.group(1))

    # 尝试查找 "The answer is X" 格式
    answer_match = re.search(r'(?:the\s+)?answer\s+is\s*:?\s*(-?[\d,]+(?:\.\d+)?)', text, re.IGNORECASE)
    if answer_match:
        return normalize_number(answer_match.group(1))

    # 最后 fallback
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return normalize_number(numbers[-1])

    return ""


def verify_answer(prediction: str, ground_truth: str) -> bool:
    """验证预测答案是否正确"""
    pred_answer = extract_answer(prediction)
    true_answer = extract_answer(ground_truth)
    return pred_answer == true_answer


def load_gsm8k_questions(data_path: str) -> list:
    """加载 GSM8K 问题"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'question': item['question'],
                'answer': item['answer'],
            })
    return data


class ExpertIterationTrainer:
    """专家迭代训练器"""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        num_iterations: int = 3,
        samples_per_iteration: int = 500,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        learning_rate: float = 2e-5,
        num_epochs_per_iteration: int = 1,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_iterations = num_iterations
        self.samples_per_iteration = samples_per_iteration
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        os.makedirs(output_dir, exist_ok=True)

    def generate_response(self, prompt: str) -> str:
        """用当前模型生成回答"""
        formatted_prompt = format_alpaca_prompt(prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
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

    def generate_training_data(self, questions: list) -> list:
        """
        生成训练数据：用当前模型回答问题，保留正确答案的样本
        """
        training_data = []
        correct_count = 0

        # 随机采样问题
        import random
        sampled_questions = random.sample(questions, min(self.samples_per_iteration, len(questions)))

        for item in tqdm(sampled_questions, desc="Generating responses"):
            question = item['question']
            ground_truth = item['answer']

            # 生成回答
            response = self.generate_response(question)

            # 验证答案
            if verify_answer(response, ground_truth):
                training_data.append({
                    'prompt': question,
                    'response': response,
                })
                correct_count += 1

        accuracy = correct_count / len(sampled_questions) if sampled_questions else 0
        print(f"Generated {len(training_data)} training examples (accuracy: {accuracy:.2%})")

        return training_data, accuracy

    def train_on_data(self, training_data: list, iteration: int):
        """在新训练数据上微调模型"""
        print(f"\nIteration {iteration + 1}: Training on {len(training_data)} examples")

        # 创建数据集
        from train_sft import SFTDataset
        dataset = SFTDataset(training_data, self.tokenizer, max_length=2048)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

        # 优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        total_loss = 0

        for epoch in range(self.num_epochs_per_iteration):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Training epoch {epoch + 1}")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss

        return total_loss / self.num_epochs_per_iteration

    def run(self, train_questions: list, use_wandb: bool = True):
        """运行专家迭代"""
        if use_wandb:
            wandb.init(project="cs336-alignment", name="expert-iteration-gsm8k")

        for iteration in range(self.num_iterations):
            print(f"\n{'='*60}")
            print(f"Expert Iteration {iteration + 1}/{self.num_iterations}")
            print(f"{'='*60}")

            # 1. 生成训练数据
            training_data, accuracy = self.generate_training_data(train_questions)

            # 2. 微调模型
            avg_loss = self.train_on_data(training_data, iteration)

            # 3. 保存模型
            save_path = os.path.join(self.output_dir, f"iteration-{iteration + 1}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

            # 4. 记录结果
            if use_wandb:
                wandb.log({
                    'iteration': iteration + 1,
                    'train_examples': len(training_data),
                    'generate_accuracy': accuracy,
                    'train_loss': avg_loss,
                })

            print(f"Iteration {iteration + 1} completed. Saved to {save_path}")

        # 保存最终模型
        final_path = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        if use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration Training")
    parser.add_argument("--model-name", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-data", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/expert-iteration")
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--samples-per-iteration", type=int, default=500)
    parser.add_argument("--num-epochs-per-iteration", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # 加载数据
    print(f"Loading training questions from: {args.train_data}")
    train_questions = load_gsm8k_questions(args.train_data)
    if args.max_samples:
        train_questions = train_questions[:args.max_samples]
    print(f"Loaded {len(train_questions)} questions")

    # 创建训练器
    trainer = ExpertIterationTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
        samples_per_iteration=args.samples_per_iteration,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        num_epochs_per_iteration=args.num_epochs_per_iteration,
        batch_size=args.batch_size,
    )

    # 运行
    trainer.run(train_questions, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
