"""
GRPO (Group Relative Policy Optimization) 训练脚本

用于在云端 GPU 上运行 Qwen2.5-Math-1.5B 的 GRPO 训练。
使用 GSM8K 作为训练数据，答案正确性作为 reward。
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb


def format_alpaca_prompt(instruction: str) -> str:
    """使用 Alpaca 模板格式化 prompt（不包含 response）"""
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
        num_str = num_str.strip().replace(',', '')
        num = float(num_str)
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        return num_str.strip()


def extract_answer(text: str) -> str:
    """从文本中提取答案（查找 #### 后面的数字或文本）"""
    # 查找 #### 后面的内容
    match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if match:
        return normalize_number(match.group(1))
    # 如果没有 ####，返回最后一个数字
    numbers = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    if numbers:
        return normalize_number(numbers[-1])
    return ""


def gsm8k_reward_fn(prediction: str, ground_truth: str) -> dict:
    """
    GSM8K 奖励函数
    返回 dict 包含 reward, format_reward, answer_reward
    """
    pred_answer = extract_answer(prediction)
    true_answer = extract_answer(ground_truth)

    # 答案正确性奖励
    answer_reward = 1.0 if pred_answer == true_answer else 0.0

    # 格式奖励（检查是否有 ####）
    format_reward = 0.5 if "####" in prediction else 0.0

    # 总奖励
    reward = answer_reward + format_reward

    return {
        "reward": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }


def load_gsm8k_questions(data_path: str, max_samples: int = None):
    """加载 GSM8K 问题（只返回 question 和 answer）"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            data.append({
                'question': item['question'],
                'answer': item['answer'],
            })
    return data


class GRPOTrainer:
    """GRPO 训练器"""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        group_size: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        learning_rate: float = 1e-6,
        batch_size: int = 4,
        num_epochs: int = 3,
        beta: float = 0.1,
        cliprange: float = 0.2,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta = beta
        self.cliprange = cliprange

        # 加载模型和 tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 创建 reference model（固定权重）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.ref_model.config.pad_token_id = self.tokenizer.pad_token_id

        # 优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        os.makedirs(output_dir, exist_ok=True)

    def generate_rollouts(self, prompts: list[str]) -> list[list[str]]:
        """
        为每个 prompt 生成 group_size 个 rollouts

        Returns:
            list of list: 每个 prompt 对应 group_size 个生成的 response
        """
        all_rollouts = []

        for prompt in prompts:
            formatted_prompt = format_alpaca_prompt(prompt)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            rollouts = []
            for _ in range(self.group_size):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # 解码生成的文本（去掉 prompt 部分）
                generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                rollouts.append(generated_text)

            all_rollouts.append(rollouts)

        return all_rollouts

    def compute_log_probs(self, model, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """
        计算给定 prompt 和 response 的 log probabilities

        Args:
            model: 语言模型
            prompts: prompt 列表
            responses: response 列表（与 prompts 一一对应）

        Returns:
            log_probs: tensor of shape (batch_size,)
        """
        log_probs_list = []

        for prompt, response in zip(prompts, responses):
            formatted_prompt = format_alpaca_prompt(prompt)
            full_text = formatted_prompt + response

            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_length = prompt_inputs['input_ids'].shape[1]

            with torch.no_grad() if model == self.ref_model else torch.enable_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # 计算 log softmax
                log_probs = torch.log_softmax(logits, dim=-1)

                # 获取 target tokens 的 log probs（shift by 1）
                target_ids = inputs['input_ids'][:, 1:]
                log_probs = log_probs[:, :-1, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                # 只取 response 部分（跳过 prompt）
                response_log_probs = log_probs[:, prompt_length-1:]

                # 求和作为序列的 log prob
                seq_log_prob = response_log_probs.sum()
                log_probs_list.append(seq_log_prob)

        return torch.stack(log_probs_list)

    def compute_group_normalized_rewards(self, rewards: list[float]) -> list[float]:
        """
        计算组归一化优势

        Args:
            rewards: list of rewards for a group

        Returns:
            advantages: normalized advantages
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8

        advantages = (rewards_tensor - mean_reward) / std_reward
        return advantages.tolist()

    def train_step(self, prompts: list[str], ground_truths: list[str]) -> dict:
        """
        执行一个 GRPO 训练步骤

        Args:
            prompts: list of questions
            ground_truths: list of ground truth answers

        Returns:
            metrics dict
        """
        # 1. 生成 rollouts
        rollouts = self.generate_rollouts(prompts)

        # 2. 计算 rewards
        all_rewards = []
        all_rollout_prompts = []
        all_rollout_responses = []

        for prompt, ground_truth, prompt_rollouts in zip(prompts, ground_truths, rollouts):
            for rollout in prompt_rollouts:
                reward_info = gsm8k_reward_fn(rollout, ground_truth)
                all_rewards.append(reward_info['reward'])
                all_rollout_prompts.append(prompt)
                all_rollout_responses.append(rollout)

        # 3. 计算 group-normalized advantages
        all_advantages = []
        for i in range(0, len(all_rewards), self.group_size):
            group_rewards = all_rewards[i:i+self.group_size]
            group_advantages = self.compute_group_normalized_rewards(group_rewards)
            all_advantages.extend(group_advantages)

        # 4. 计算 old policy 的 log probs
        old_log_probs = self.compute_log_probs(self.model, all_rollout_prompts, all_rollout_responses)

        # 5. 计算 reference model 的 log probs
        ref_log_probs = self.compute_log_probs(self.ref_model, all_rollout_prompts, all_rollout_responses)

        # 6. GRPO loss 更新（简化版，多次迭代）
        advantages_tensor = torch.tensor(all_advantages, device=self.model.device)

        # 计算 KL penalty
        kl_penalty = self.beta * (old_log_probs - ref_log_probs)

        # 总 reward 包含 KL penalty
        advantages_with_kl = advantages_tensor - kl_penalty.detach()

        # 简单 policy gradient loss（可以用更复杂的 clip loss）
        loss = -(old_log_probs * advantages_with_kl).mean()

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'mean_reward': sum(all_rewards) / len(all_rewards),
            'mean_advantage': sum(all_advantages) / len(all_advantages),
        }

    def train(self, train_data: list, use_wandb: bool = True):
        """主训练循环"""
        if use_wandb:
            wandb.init(project="cs336-alignment", name="grpo-gsm8k")

        num_batches = len(train_data) // self.batch_size

        for epoch in range(self.num_epochs):
            epoch_metrics = {'loss': 0, 'mean_reward': 0, 'mean_advantage': 0}

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch_idx in progress_bar:
                # 获取 batch 数据
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_data = train_data[batch_start:batch_end]

                prompts = [item['question'] for item in batch_data]
                ground_truths = [item['answer'] for item in batch_data]

                # 执行训练步骤
                metrics = self.train_step(prompts, ground_truths)

                # 更新统计
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]

                # 更新进度条
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

                # 记录 wandb
                if use_wandb and batch_idx % 10 == 0:
                    wandb.log({**metrics, 'epoch': epoch, 'step': batch_idx})

            # 计算 epoch 平均
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            print(f"Epoch {epoch+1} completed: {epoch_metrics}")

            # 保存 checkpoint
            save_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

        # 保存最终模型
        final_path = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)

        if use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Qwen2.5-Math")
    parser.add_argument("--model-name", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train-data", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # 加载数据
    print(f"Loading training data from: {args.train_data}")
    train_data = load_gsm8k_questions(args.train_data, max_samples=args.max_samples)
    print(f"Loaded {len(train_data)} training examples")

    # 创建 trainer
    trainer = GRPOTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        group_size=args.group_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        beta=args.beta,
    )

    # 开始训练
    trainer.train(train_data, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
