"""
GRPO (Group Relative Policy Optimization) 训练脚本 - MetaMathQA 版本

基于 EI 模型，在 MetaMathQA 上使用 r1_zero 格式和奖励函数进行 GRPO 训练。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 导入 drgrpo_grader
sys.path.insert(0, str(Path(__file__).parent.parent / "cs336_alignment"))
from drgrpo_grader import r1_zero_reward_fn, extract_answer


def format_r1_zero_prompt(question: str) -> str:
    """使用 r1_zero prompt 模板格式化问题"""
    template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e. <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""
    return template.format(question=question)


def load_metamath_data(data_path: str, max_samples: int = None):
    """加载 MetaMathQA 数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            # 从 response 中提取 ground_truth 答案
            ground_truth = extract_answer(item['response'])
            if not ground_truth:
                # 备用方案：尝试从 #### 后提取
                import re
                match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', item['response'])
                if match:
                    ground_truth = match.group(1).replace(',', '')

            data.append({
                'question': item['query'],
                'ground_truth': ground_truth,
            })
    return data


class GRPOTrainerMetaMath:
    """MetaMathQA 专用 GRPO 训练器"""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        checkpoint_dir: str = None,
        group_size: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        learning_rate: float = 1e-6,
        batch_size: int = 1,
        num_epochs: int = 3,
        beta: float = 0.1,
        cliprange: float = 0.2,
        gradient_accumulation_steps: int = 8,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir or output_dir
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.beta = beta
        self.cliprange = cliprange
        self.gradient_accumulation_steps = gradient_accumulation_steps

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 加载模型，启用 gradient checkpointing 节省显存
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        # 创建 reference model（固定权重）
        print("Loading reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
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
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def generate_rollouts(self, questions: list[str]) -> list[list[str]]:
        """为每个问题生成 group_size 个 rollouts"""
        all_rollouts = []

        for question in questions:
            prompt = format_r1_zero_prompt(question)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
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

                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                # 补上 prompt 缺失的部分，形成完整 response
                full_response = "<think>" + generated_text
                rollouts.append(full_response)

            all_rollouts.append(rollouts)

        return all_rollouts

    def compute_rewards(self, rollouts: list[list[str]], ground_truths: list[str]) -> tuple[list[float], list[dict]]:
        """计算奖励值"""
        all_rewards = []
        reward_infos = []

        for prompt_rollouts, gt in zip(rollouts, ground_truths):
            for rollout in prompt_rollouts:
                reward_info = r1_zero_reward_fn(rollout, gt, fast=True)
                all_rewards.append(reward_info['reward'])
                reward_infos.append(reward_info)

        return all_rewards, reward_infos

    def compute_group_normalized_advantages(self, rewards: list[float]) -> list[float]:
        """计算组归一化优势"""
        all_advantages = []

        for i in range(0, len(rewards), self.group_size):
            group_rewards = rewards[i:i + self.group_size]
            rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32)
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_reward) / std_reward
            all_advantages.extend(advantages.tolist())

        return all_advantages

    def compute_log_probs(self, model, questions: list[str], responses: list[str]) -> torch.Tensor:
        """计算 log probabilities"""
        log_probs_list = []

        for question, response in zip(questions, responses):
            prompt = format_r1_zero_prompt(question)
            full_text = prompt + response[len("<think>"):]

            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_length = prompt_inputs['input_ids'].shape[1]

            with torch.no_grad() if model == self.ref_model else torch.enable_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                log_probs = torch.log_softmax(logits, dim=-1)
                target_ids = inputs['input_ids'][:, 1:]
                log_probs = log_probs[:, :-1, :].gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                # 只取 response 部分
                response_log_probs = log_probs[:, prompt_length - 1:]
                seq_log_prob = response_log_probs.sum()
                log_probs_list.append(seq_log_prob)

        return torch.stack(log_probs_list)

    def train_step(self, questions: list[str], ground_truths: list[str], step: int) -> dict:
        """执行一个 GRPO 训练步骤"""
        # 1. 生成 rollouts
        rollouts = self.generate_rollouts(questions)

        # 2. 计算 rewards
        all_rewards, reward_infos = self.compute_rewards(rollouts, ground_truths)

        # 3. 计算 group-normalized advantages
        all_advantages = self.compute_group_normalized_advantages(all_rewards)

        # 4. 准备训练数据（扩展 questions 和 responses 以匹配 rollouts）
        all_questions = []
        all_responses = []
        for question, prompt_rollouts in zip(questions, rollouts):
            all_questions.extend([question] * len(prompt_rollouts))
            all_responses.extend(prompt_rollouts)

        # 5. 计算 old policy 和 reference model 的 log probs
        old_log_probs = self.compute_log_probs(self.model, all_questions, all_responses)
        ref_log_probs = self.compute_log_probs(self.ref_model, all_questions, all_responses)

        # 6. 计算 loss（支持梯度累积）
        advantages_tensor = torch.tensor(all_advantages, device=self.model.device)

        # KL penalty
        kl_penalty = self.beta * (old_log_probs - ref_log_probs)
        advantages_with_kl = advantages_tensor - kl_penalty.detach()

        # Policy gradient loss
        loss = -(old_log_probs * advantages_with_kl).mean()
        loss = loss / self.gradient_accumulation_steps

        # 反向传播
        loss.backward()

        metrics = {
            'loss': loss.item() * self.gradient_accumulation_steps,
            'mean_reward': sum(all_rewards) / len(all_rewards),
            'mean_advantage': sum(all_advantages) / len(all_advantages),
            'format_accuracy': sum(1 for r in reward_infos if r['format_reward'] > 0) / len(reward_infos),
            'answer_accuracy': sum(1 for r in reward_infos if r['answer_reward'] > 0) / len(reward_infos),
        }

        # 梯度累积步数达到后更新
        if (step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return metrics

    def train(self, train_data: list, use_wandb: bool = True):
        """主训练循环"""
        if use_wandb:
            try:
                import swanlab as wandb
                wandb.init(project="cs336-alignment", name="grpo-metamath")
            except ImportError:
                import wandb
                wandb.init(project="cs336-alignment", name="grpo-metamath")

        num_batches = len(train_data) // self.batch_size

        self.optimizer.zero_grad()

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # 打乱数据
            import random
            random.shuffle(train_data)

            epoch_metrics = {
                'loss': 0, 'mean_reward': 0, 'mean_advantage': 0,
                'format_accuracy': 0, 'answer_accuracy': 0
            }

            progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}")

            for batch_idx in progress_bar:
                batch_start = batch_idx * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_data = train_data[batch_start:batch_end]

                questions = [item['question'] for item in batch_data]
                ground_truths = [item['ground_truth'] for item in batch_data]

                # 执行训练步骤
                global_step = epoch * num_batches + batch_idx
                metrics = self.train_step(questions, ground_truths, global_step)

                # 更新统计
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]

                # 更新进度条
                progress_bar.set_postfix({
                    'reward': f"{metrics['mean_reward']:.3f}",
                    'format': f"{metrics['format_accuracy']:.2%}",
                    'ans': f"{metrics['answer_accuracy']:.2%}",
                })

                # 记录 wandb
                if use_wandb and batch_idx % 10 == 0:
                    import swanlab as wandb_module
                    wandb_module.log({**metrics, 'epoch': epoch, 'step': global_step})

            # 计算 epoch 平均
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

            print(f"Epoch {epoch + 1} 平均指标: {epoch_metrics}")

            # 保存 checkpoint 到数据盘
            save_path = os.path.join(self.checkpoint_dir, f"checkpoint-epoch-{epoch + 1}")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Checkpoint saved to {save_path}")

        # 保存最终模型到输出目录
        final_path = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        print(f"Final model saved to {final_path}")

        if use_wandb:
            import swanlab as wandb_module
            wandb_module.finish()


def main():
    parser = argparse.ArgumentParser(description="GRPO Training on MetaMathQA")
    parser.add_argument("--model-name", type=str, default="models/ei_metamath")
    parser.add_argument("--train-data", type=str, default="data/MetaMathQA/train.jsonl")
    parser.add_argument("--output-dir", type=str, default="models/grpo_metamath")
    parser.add_argument("--checkpoint-dir", type=str, default="/root/autodl-tmp/grpo_checkpoints")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    # 加载数据
    print(f"Loading training data from: {args.train_data}")
    train_data = load_metamath_data(args.train_data, max_samples=args.max_samples)
    print(f"Loaded {len(train_data)} training examples")

    # 创建 trainer
    trainer = GRPOTrainerMetaMath(
        model_name=args.model_name,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        group_size=args.group_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        beta=args.beta,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # 开始训练
    trainer.train(train_data, use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
