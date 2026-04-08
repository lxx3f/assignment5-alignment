import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    对提示词进行分词并输出字符串，同时构建掩码：回复token对应的位置设为1，
    其余token（提示词或填充位）对应的位置设为0。

    Args:
        prompt_strs: list[str], 提示词列表
        output_strs: list[str], 输出字符串列表
        tokenizer: 模型分词器

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": 经过分词处理的提示词与输出字符串
            "labels": 标签（实际是移位1的input_ids）
            "response_mask": 标签掩码，回复token对应位置为1，其余为0
    """
    assert len(prompt_strs) == len(output_strs), "prompt_strs and output_strs must have the same length"
    batch_size = len(prompt_strs)
    # 对提示词进行分词
    prompt_tokens = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    prompt_ids_list = prompt_tokens["input_ids"]
    # 对输出字符串进行分词
    output_tokens = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    output_ids_list = output_tokens["input_ids"]
    # 组合提示词和输出字符串的token id
    prompt_and_output_ids = [
        prompt_ids + output_ids for prompt_ids, output_ids in zip(prompt_ids_list, output_ids_list)
    ]
    # 增加掩码
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(ids) for ids in prompt_and_output_ids)
    full_padded_list = torch.full(
        (batch_size, max_len),
        pad_id,
        dtype=torch.long
    ) # shape: (batch_size, max_len)
    for i, ids in enumerate(prompt_and_output_ids):
        full_padded_list[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    
    # 创建标签
    input_ids = full_padded_list[:, :-1].contiguous()
    labels = full_padded_list[:, 1:].contiguous()

    # 创建response掩码
    response_mask = torch.zeros_like(labels, dtype=torch.long)
    for i in range(batch_size):
        prompt_len = len(prompt_ids_list[i])
        output_len = len(output_ids_list[i])
        if output_len == 0:
            continue
        
        start = max(prompt_len-1,0)
        end = min(prompt_len + output_len - 1, labels.size(1))
        if end > start:
            response_mask[i, start:end] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

def compute_entropy(logits):
    """
    计算 next-token predictions 的熵（即在 vocabulary 维度上的熵）

    使用数值稳定的实现方式（log_softmax）。

    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
            包含未归一化的 logits。

    Returns:
        torch.Tensor of shape (batch_size, sequence_length)
            每个 next-token prediction 的熵。
    """
    # 使用 log_softmax 获取 log probabilities（数值稳定）
    log_probs = F.log_softmax(logits, dim=-1)
    # 计算 probabilities
    probs = torch.exp(log_probs)
    # 计算熵: H(p) = -sum_x p(x) * log p(x)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    # entropy 的 shape: (batch_size, sequence_length)
    return entropy

def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    """
    获取 response 的条件 log-probabilities，以及可选的 token entropy。

    Args:
        model: PreTrainedModel，用于评分的模型（应在正确的 device 上，
            如果不需要计算梯度则应处于 inference 模式）。
        input_ids: torch.Tensor of shape (batch_size, sequence_length)，
            由 tokenization 方法生成的 prompt + response tokens。
        labels: torch.Tensor of shape (batch_size, sequence_length)，
            由 tokenization 方法生成的 labels（shifted input_ids）。
        return_token_entropy: bool，是否返回 next-token predictions 的 entropy。

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length)，
                给定 prompt 的 response 的条件 log-probabilities。
                注意：我们还没有 mask 掉 prompt 和 padding 对应的 token indices，
                这会在训练循环中完成。
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length)，
                next-token predictions 的 entropy。与 log_probs 一样，
                我们还没有 mask 掉 prompt 和 padding 对应的 token indices。
    """
    # 获取 logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # 计算 log probabilities（在 vocab 维度上做 log_softmax）
    # shape: (batch_size, seq_len, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)

    # 获取 labels 对应的 log probs
    # labels 是目标 token IDs，我们需要在 vocab 维度上 gather
    # log_probs_all 的形状是 (batch_size, seq_len, vocab_size)
    # labels 的形状是 (batch_size, seq_len)
    # 我们需要将 labels 扩展一个维度以便 gather
    batch_size, seq_len, vocab_size = log_probs_all.shape
    labels_expanded = labels.unsqueeze(-1)  # (batch_size, seq_len, 1)

    # gather 对应的 log probs
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_expanded).squeeze(-1)

    result = {"log_probs": log_probs}

    # 如果需要，计算 entropy
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result


def masked_normalize(tensor, mask, normalize_constant=1.0, dim=None):
    """
    对 tensor 在指定维度上求和并按常数归一化，只考虑 mask == 1 的元素。

    Args:
        tensor: torch.Tensor，要操作的张量。
        mask: torch.Tensor，与 tensor 形状相同；值为 1 的位置参与求和。
        normalize_constant: float，用于归一化的常数。
        dim: int | None，求和的维度。如果为 None，则对所有非 mask 元素求和。

    Returns:
        torch.Tensor，归一化后的和，mask 元素（mask == 0）不计入求和。
    """
    # 将 mask 转换为与 tensor 相同的类型，用于乘法
    mask_float = mask.to(tensor.dtype)

    # 先对 mask 后的 tensor 求和
    masked_sum = (tensor * mask_float).sum(dim=dim)

    # 归一化
    result = masked_sum / normalize_constant

    return result



def sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant=1.0):
    """
    执行一个 microbatch 的前向-后向传播。

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length)，
            来自 SFT policy 的 per-token log-probabilities。
        response_mask: torch.Tensor of shape (batch_size, sequence_length)，
            1 表示 response tokens，0 表示 prompt/padding。
        gradient_accumulation_steps: int，每个 optimizer step 的 microbatch 数量。
        normalize_constant: float，用于归一化的常数。

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            loss: scalar tensor。microbatch 的 loss，已针对梯度累积调整。
            metadata: dict，包含底层 loss 调用的元数据，以及你可能想要记录的其他统计信息。
    """
    # SFT loss: 对 response tokens 的 negative log-likelihood 求平均
    # loss = -sum(log_probs * mask) / (normalize_constant)
    # 注意：需要除以 gradient_accumulation_steps 以正确处理梯度累积

    # 将 mask 转换为与 policy_log_probs 相同的 dtype
    # mask_float = response_mask.to(policy_log_probs.dtype)

    # 计算 negative log-likelihood（注意 policy_log_probs 是 log probs，需要取负）
    # 对于 SFT，目标是最小化 -log p(response|prompt)
    # policy_log_probs 是给定 label 的 log prob，所以直接取负并 mask
    # neg_log_likelihood = -policy_log_probs * mask_float
    neg_log_likelihood = -policy_log_probs

    # 对 sequence 维度求和
    # seq_sum = neg_log_likelihood.sum(dim=-1)
    seq_sum = masked_normalize(neg_log_likelihood, response_mask, normalize_constant, dim=-1)

    # 对 batch 维度求平均
    loss = seq_sum.mean()

    # # 归一化（如果有）
    # if normalize_constant != 1.0:
    #     loss = loss / normalize_constant

    # 针对梯度累积调整 loss
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps

    # 执行反向传播
    loss.backward()

    # 元数据
    metadata = {
        "loss": loss.item(),
    }

    return loss, metadata


