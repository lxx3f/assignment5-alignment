# GRPO 训练踩坑记录

## 问题 1: `is_correct` 未定义错误

**报错信息：**
```
cannot access local variable 'is_correct' where it is not associated with a value
```

**原因：**
`r1_zero_reward_fn` 函数中，当 `model_answer` 不包含 `\boxed` 时，代码直接 continue，导致后续使用 `is_correct` 时变量未定义。

**修复：**
```python
# 在 drgrpo_grader.py 中
# 1. 添加 is_correct = False 初始化
# 2. 处理无 \boxed 的情况（返回 format_reward=1, answer_reward=0）
```

---

## 问题 2: 答案正确率始终为 0% (ans=0.00%)

**现象：**
- `format=100%`（格式正确）
- `ans=0%`（答案错误）
- 平均奖励始终为 0

**根因：**
- **Prompt 要求**：`<answer> answer here </answer>`
- **奖励函数要求**：`<answer>\boxed{123}</answer>`
- **EI 模型实际输出**：`<answer>123</answer>`（无 `\boxed`）

由于格式不匹配，奖励函数把无 `\boxed` 的答案判定为错误。

**修复：**
修改 `r1_zero_reward_fn`，允许无 `\boxed` 的答案格式：
```python
if "\\boxed" in model_answer:
    model_answer = extract_answer(model_answer)
    # ... 处理 boxed 答案
# 不再 else return 0，而是直接使用 model_answer.strip()
```

---

## 问题 3: MetaMathQA 数据格式不统一

**统计：**
- `\boxed{}` 格式：~30%
- `#### 123` 格式：~60%
- `The answer is: X` 格式：~9.4%（无法提取 ground_truth）

**影响：**
约 9.4% 的样本 ground_truth 为 None，这些样本会奖励为 0，但对训练影响有限。

---

## Debug 方法

1. **检查奖励函数逻辑**
   ```python
   from drgrpo_grader import r1_zero_reward_fn
   result = r1_zero_reward_fn(response, ground_truth)
   print(result)  # 验证奖励计算是否正确
   ```

2. **检查数据格式**
   ```python
   # 统计各种答案格式占比
   # 确认 ground_truth 提取是否正常
   ```

3. **检查 Prompt 与奖励函数匹配**
   - Prompt 模板要求什么格式？
   - 奖励函数检查什么格式？
   - 模型实际输出什么格式？

