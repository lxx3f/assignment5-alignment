import torch
from typing import Callable, List, Tuple, Dict, Optional


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """
    计算 tensor 在指定维度上的均值，只考虑 mask == 1 的元素。

    Args:
        tensor: torch.Tensor，要计算均值的张量。
        mask: torch.Tensor，与 tensor 形状相同；值为 1 的位置参与计算。
        dim: int | None，要计算均值的维度。如果为 None，则对所有非 mask 元素求平均。

    Returns:
        torch.Tensor，指定维度上的均值，只考虑 mask 元素。
    """
    # 将 mask 转换为与 tensor 相同的类型
    mask_float = mask.to(tensor.dtype)

    # 计算 masked 元素的和
    masked_sum = (tensor * mask_float).sum(dim=dim)

    # 计算 mask 的元素个数
    mask_count = mask_float.sum(dim=dim)

    # 计算均值，对于 mask_count 为 0 的位置会得到 nan
    mean = masked_sum / mask_count

    return mean


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    计算每组回复的奖励，并按组规模进行归一化处理。

    Args:
        reward_fn (Callable[[str, str], Dict[str, float]]): 对照真实标准答案对输出回复进行打分，
            生成一个包含"reward", "format_reward", "answer_reward"键的字典。
        rollout_responses (List[str]): 回复列表。len=batch_size=prompt_num*group_size
        repeated_ground_truths (List[str]): 这些示例的真实标签。len=batch_size=prompt_num*group_size
        group_size (int): 每组回复的个数。
        advantage_eps (float, optional): 防止除零的归一化常数，默认为1e-8。
        normalize_by_std (bool, optional): 是否按标准差进行归一化处理，默认为False。
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]: 奖励值归一化值，原始奖励值，额外信息。
            normalized_rewards (torch.Tensor): 奖励值归一化值。 shape=(batch_size,)
            raw_rewards (torch.Tensor): 奖励值。 shape=(batch_size,)
            metadata (Dict[str, float]): 额外的信息。
    """
    # 1. 计算所有 rollout 的原始奖励
    raw_rewards = []
    format_rewards = []
    answer_rewards = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards.append(reward_dict["reward"])
        # 用get默认值，避免KeyError, 因为fromat_reward和answer_reward并不是绝对必需的，后面的计算只用到了reward
        format_rewards.append(reward_dict.get("format_reward", reward_dict["reward"]))
        answer_rewards.append(reward_dict.get("answer_reward", reward_dict["reward"]))

    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)

    # 2. 按 group 分组计算优势值
    num_groups = len(rollout_responses) // group_size
    normalized_rewards = []

    for i in range(num_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = raw_rewards_tensor[start_idx:end_idx]

        # 计算组内均值
        group_mean = group_rewards.mean()

        # 计算组内标准差（如果 normalize_by_std 为 True）
        if normalize_by_std:
            group_std = group_rewards.std()
            # 避免除以零
            denominator = group_std + advantage_eps
        else:
            # 如果不按 std 归一化，denominator 为 1
            denominator = 1.0

        # 计算组归一化奖励（优势值）
        group_advantages = (group_rewards - group_mean) / denominator
        normalized_rewards.append(group_advantages)

    # 拼接所有组的优势值
    normalized_rewards_tensor = torch.cat(normalized_rewards)

    # 3. 准备元数据
    metadata = {
        "mean_reward": raw_rewards_tensor.mean().item(),
        "std_reward": raw_rewards_tensor.std().item(),
        "mean_format_reward": sum(format_rewards) / len(format_rewards),
        "mean_answer_reward": sum(answer_rewards) / len(answer_rewards),
    }

    return normalized_rewards_tensor, raw_rewards_tensor, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算朴素策略梯度损失（Naive Policy Gradient Loss）。

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1)，
            每个 rollout response 的原始奖励或优势值。
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length)，
            策略的 per-token log-probabilities。

    Returns:
        torch.Tensor of shape (batch_size, sequence_length)：
            每个 token 的策略梯度损失。
    """
    # 将 rewards 扩展到与 policy_log_probs 相同的形状
    # raw_rewards_or_advantages: (batch_size, 1) -> (batch_size, 1) 用于广播
    rewards = raw_rewards_or_advantages

    # 计算损失: -log_probs * rewards
    # 广播机制会自动将 rewards 扩展到 sequence_length 维度
    loss = -policy_log_probs * rewards

    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GRPO-Clip 损失。

    结合 PPO 的 clipping 机制来限制策略更新幅度：
    L^CLIP = -min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    其中 r_t = exp(policy_log_probs - old_log_probs)

    Args:
        advantages: torch.Tensor of shape (batch_size, 1)，
            每个 rollout response 的优势值。
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length)，
            新策略的 per-token log-probabilities。
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length)，
            旧策略的 per-token log-probabilities。
        cliprange: float，裁剪范围参数。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: torch.Tensor of shape (batch_size, sequence_length)，
                GRPO-Clip per-token loss。
            metadata: dict，包含 clip fraction 等信息。
    """
    # 计算 importance ratio: r_t = exp(log π_θ - log π_old)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # 计算 clipped ratio
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # 扩展 advantages 到与 ratio 相同的形状 (batch_size, seq_len)
    advantages_expanded = advantages.expand_as(ratio)

    # 计算两种 objective
    objective_unclipped = ratio * advantages_expanded
    objective_clipped = ratio_clipped * advantages_expanded

    # 取最小值（防止策略更新过大）
    objective = torch.min(objective_unclipped, objective_clipped)

    # 损失是负的 objective（因为 PyTorch 默认最小化）
    loss = -objective

    # 计算 clip fraction（用于监控）
    clip_fraction = (torch.abs(ratio - 1.0) > cliprange).float().mean()

    metadata = {
        "clip_fraction": clip_fraction,
    }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    策略梯度损失的包装器函数，根据 loss_type 调用不同的实现。

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length)，
            策略的 per-token log-probabilities。
        loss_type: str，损失函数类型，可选 "no_baseline", "reinforce_with_baseline", "grpo_clip"。
        raw_rewards: torch.Tensor of shape (batch_size, 1)，原始奖励。
        advantages: torch.Tensor of shape (batch_size, 1)，优势值。
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length)，
            旧策略的 per-token log-probabilities。
        cliprange: float，裁剪范围（仅用于 grpo_clip）。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 损失和元数据。
    """
    if loss_type == "no_baseline":
        # 使用原始奖励，无 baseline
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
        return loss, metadata

    elif loss_type == "reinforce_with_baseline":
        # 使用优势值（已包含 baseline）
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
        return loss, metadata

    elif loss_type == "grpo_clip":
        # 使用 GRPO-Clip
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        return loss, metadata

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Must be one of: no_baseline, reinforce_with_baseline, grpo_clip")


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行 GRPO microbatch 的前向-后向传播。

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length)，
            策略的 per-token log-probabilities。
        response_mask: torch.Tensor of shape (batch_size, sequence_length)，
            1 表示 response tokens，0 表示 prompt/padding。
        gradient_accumulation_steps: int，梯度累积步数。
        loss_type: str，损失函数类型。
        raw_rewards: torch.Tensor | None，原始奖励（用于 no_baseline）。
        advantages: torch.Tensor | None，优势值（用于 reinforce_with_baseline 和 grpo_clip）。
        old_log_probs: torch.Tensor | None，旧策略 log probs（用于 grpo_clip）。
        cliprange: float | None，裁剪范围（用于 grpo_clip）。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: loss 和 metadata。
    """
    # 1. 计算 policy gradient loss
    loss_per_token, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards if raw_rewards is not None else torch.zeros_like(advantages),
        advantages=advantages if advantages is not None else torch.zeros_like(raw_rewards),
        old_log_probs=old_log_probs if old_log_probs is not None else policy_log_probs,
        cliprange=cliprange if cliprange is not None else 0.2,
    )

    # 2. 应用 response_mask（只计算 response tokens 的 loss）
    mask_float = response_mask.to(loss_per_token.dtype)
    masked_loss = loss_per_token * mask_float

    # 3. 计算每个 sequence 的 mask 数量
    mask_count = response_mask.sum(dim=-1).float()  # (batch_size,)
    mask_count = torch.clamp(mask_count, min=1.0)  # 避免除以零

    # 4. 对每个 sequence 求和，然后除以该 sequence 的 mask 数量
    seq_sum = masked_loss.sum(dim=-1)  # (batch_size,)
    seq_mean = seq_sum / mask_count  # (batch_size,)

    # 5. 对 batch 求平均
    loss = seq_mean.mean()  # scalar

    # 6. 考虑梯度累积
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps

    # 5. 执行反向传播
    loss.backward()

    return loss, metadata