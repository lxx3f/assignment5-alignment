import torch
import torch.nn.functional as F


# Alpaca prompt 模板
ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    计算单个偏好对的 DPO (Direct Preference Optimization) 损失。

    使用与 SFT 阶段一致的 Alpaca 模板格式化 prompt 和 response，
    并在末尾添加 EOS token。

    Args:
        lm: 正在训练的语言模型 (策略模型 π_θ)
        lm_ref: 参考语言模型 (π_ref)
        tokenizer: 分词器
        beta: DPO beta 超参数
        prompt: 提示文本
        response_chosen: 偏好的响应 (y_w)
        response_rejected: 非偏好的响应 (y_l)

    Returns:
        torch.Tensor: 该样本的 DPO 损失值
    """
    # 读取设备
    device = next(lm.parameters()).device

    # 1. 使用 Alpaca 模板构造完整文本，并添加 EOS
    chosen_full = ALPACA_TEMPLATE.format(instruction=prompt, response=response_chosen)
    rejected_full = ALPACA_TEMPLATE.format(instruction=prompt, response=response_rejected)

    # 添加 EOS token
    chosen_full += tokenizer.eos_token
    rejected_full += tokenizer.eos_token

    # 同时构造 prompt 部分（用于确定 response 起始位置）
    prompt_only = ALPACA_TEMPLATE.format(instruction=prompt, response="")

    # 2. Tokenize
    chosen_tokens = tokenizer(chosen_full, return_tensors="pt")
    rejected_tokens = tokenizer(rejected_full, return_tensors="pt")
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt")

    chosen_input_ids = chosen_tokens["input_ids"].to(device)
    rejected_input_ids = rejected_tokens["input_ids"].to(device)
    prompt_length = prompt_tokens["input_ids"].shape[1]

    # 3. 计算策略模型和参考模型的 log probs
    # with torch.no_grad():
    #     chosen_logits = lm(chosen_input_ids).logits
    #     rejected_logits = lm(rejected_input_ids).logits

    # 策略模型：需要梯度
    chosen_logits = lm(chosen_input_ids).logits
    rejected_logits = lm(rejected_input_ids).logits

    # 参考模型：不需要梯度
    with torch.no_grad():
        chosen_ref_logits = lm_ref(chosen_input_ids).logits
        rejected_ref_logits = lm_ref(rejected_input_ids).logits

    chosen_log_probs = torch.log_softmax(chosen_logits, dim=-1)
    rejected_log_probs = torch.log_softmax(rejected_logits, dim=-1)
    chosen_ref_log_probs = torch.log_softmax(chosen_ref_logits, dim=-1)
    rejected_ref_log_probs = torch.log_softmax(rejected_ref_logits, dim=-1)

    chosen_log_prob = chosen_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_log_prob = rejected_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    chosen_ref_log_prob = chosen_ref_log_probs[:, :-1, :].gather(-1, chosen_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)
    rejected_ref_log_prob = rejected_ref_log_probs[:, :-1, :].gather(-1, rejected_input_ids[:, 1:, None]).squeeze(-1).sum(dim=-1)


    # 4. 计算 DPO 损失
    chosen_ratio = chosen_log_prob - chosen_ref_log_prob
    rejected_ratio = rejected_log_prob - rejected_ref_log_prob

    # β * (chosen_ratio - rejected_ratio)
    logits = beta * (chosen_ratio - rejected_ratio)

    # -log σ(logits)
    loss = -F.logsigmoid(logits)

    return loss
