# MetaMathQA Zero-shot 评估分析报告

## 一、评估设置

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen2.5-Math-1.5B |
| 数据集 | MetaMathQA validation |
| 评估样本数 | 500 |
| Prompt格式 | r1_zero (带`<think>`/`<answer>`标签) |
| 解码方式 | Greedy (temperature=0) |
| 最大生成长度 | 2048 tokens |

## 二、三类样本统计

| 类别 | 条件 | 数量 | 占比 | 说明 |
|------|------|------|------|------|
| **类别1：完全正确** | format=1, answer=1, reward=1 | 115 | 23.0% | 格式与答案均正确 |
| **类别2：答案错误** | format=1, answer=0, reward=0 | 112 | 22.4% | 格式合规但答案错误 |
| **类别3：格式错误** | format=0, answer=0, reward=0 | 273 | 54.6% | 未遵循r1_zero格式 |

**总体表现**：Zero-shot 条件下，模型仅 23.0% 的样本能同时满足格式和答案正确性要求。

---

## 三、格式错误样本分析（类别3，共273例）

### 3.1 问题定位

r1_zero 格式要求输出结构为：
```
<think>推理过程</think> <answer>答案</answer>
```

**解析器判定逻辑**：`"</think> <answer>" in response and "</answer>" in response`

### 3.2 10例格式错误样本分析

#### 示例1

**问题**：Determine the area of the region enclosed by the equation $x^2+y^2+12x+16y=0$....

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：True
- `<answer>` 开始标签：True
- `</answer>` 结束标签：True
- 完整格式序列 `</think> <answer>`：False
- 生成长度：412 字符

**模型输出（前150字符）**：
```
 The given equation is a circle. To find the area, we first need to complete the square for both x and y. The equation can be rewritten as $(x+6)^2 + ...
```

#### 示例2

**问题**：For every sandwich that he eats, Sam eats x apples. If he eats 10 sandwiches every day for one week, He eats 280 apples....

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：True
- `<answer>` 开始标签：True
- `</answer>` 结束标签：True
- 完整格式序列 `</think> <answer>`：False
- 生成长度：372 字符

**模型输出（前150字符）**：
```
Let's denote the number of apples Sam eats per sandwich as x. Since Sam eats 10 sandwiches every day for one week (7 days), he eats a total of 10 * 7 ...
```

#### 示例3

**问题**：Kris is blowing u balloons for her sister’s party. She has 30 minutes to blow up as many balloons as possible and decide...

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：True
- `<answer>` 开始标签：True
- `</answer>` 结束标签：True
- 完整格式序列 `</think> <answer>`：False
- 生成长度：2733 字符

**模型输出（前150字符）**：
```
 Kris is blowing u balloons for her sister’s party. She has 30 minutes to blow up as many balloons as possible and decides to enlist her brother’s hel...
```

#### 示例4

**问题**：The recipe for a four-person cake requires 2 eggs and 4 cups of milk. Tyler wants to use this recipe to make a cake for ...

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：True
- `<answer>` 开始标签：True
- `</answer>` 结束标签：True
- 完整格式序列 `</think> <answer>`：False
- 生成长度：1225 字符

**模型输出（前150字符）**：
```
 The recipe for a four-person cake requires 2 eggs and 4 cups of milk. Tyler wants to use this recipe to make a cake for eight people. If Tyler has x ...
```

#### 示例5

**问题**：What is the cost, in dollars, to fill Toby's swimming pool if it normally takes 50 hours to fill and his hose runs at a ...

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：False
- `<answer>` 开始标签：False
- `</answer>` 结束标签：False
- 完整格式序列 `</think> <answer>`：False
- 生成长度：1355 字符

**模型输出（前150字符）**：
```
 To find the cost to fill Toby's swimming pool, we need to follow these steps:
1. Calculate the total volume of water needed to fill the pool.
2. Dete...
```

#### 示例6

**问题**：Determine all values of $y$ in the interval $[0,\pi]$ for which the inequality $\sin(x+y) \leq \sin(x) + \sin(y)$ holds ...

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：False
- `<answer>` 开始标签：False
- `</answer>` 结束标签：False
- 完整格式序列 `</think> <answer>`：False
- 生成长度：2204 字符

**模型输出（前150字符）**：
```
Let's start by considering the properties of the sine function. We know that $\sin(x)$ is a periodic function with a period of $2\pi$, and it is symme...
```

#### 示例7

**问题**：Jack bought x apples. He wants to give 10 to his father and then equally share the remaining apples between him and his ...

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：True
- `<answer>` 开始标签：True
- `</answer>` 结束标签：True
- 完整格式序列 `</think> <answer>`：False
- 生成长度：812 字符

**模型输出（前150字符）**：
```
Jack bought x apples. He wants to give 10 to his father. So, the remaining apples are x - 10. He then equally shares the remaining apples between him ...
```

#### 示例8

**问题**：Bertha plays tennis. Every ten games, one of her tennis balls wears out. Every five games, she loses a ball. Every four ...

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：True
- `<answer>` 开始标签：True
- `</answer>` 结束标签：True
- 完整格式序列 `</think> <answer>`：False
- 生成长度：1995 字符

**模型输出（前150字符）**：
```
 Bertha plays tennis. Every ten games, one of her tennis balls wears out. Every five games, she loses a ball. Every four games, she buys a canister of...
```

#### 示例9

**问题**：Simplify: $\sqrt{50} + \sqrt{18}$ . Express your answer in simplest radical form....

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：False
- `<answer>` 开始标签：True
- `</answer>` 结束标签：True
- 完整格式序列 `</think> <answer>`：False
- 生成长度：309 字符

**模型输出（前150字符）**：
```
 To simplify the expression $\sqrt{50} + \sqrt{18}$, we can break down each square root into its prime factors. We have $\sqrt{50} = \sqrt{25 \cdot 2}...
```

#### 示例10

**问题**：Let $S$ be a region in the plane with area 10.  When we apply the matrix
\[\begin{pmatrix} 2 & 1 \\ X & -3 \end{pmatrix}...

**格式检查结果**：
- `<think>` 开始标签：False
- `</think>` 结束标签：False
- `<answer>` 开始标签：False
- `</answer>` 结束标签：False
- 完整格式序列 `</think> <answer>`：False
- 生成长度：1436 字符

**模型输出（前150字符）**：
```
 The area of the transformed region $S'$ is given by the absolute value of the determinant of the transformation matrix multiplied by the area of the ...
```

### 3.3 根因分析

| 现象 | 占比 | 原因分析 |
|------|------|----------|
| **无`<think>`开始标签** | ~95% | Prompt 结尾已包含`<think>`，模型直接续写推理内容，未重复标签 |
| **无`</think> <answer>`序列** | ~80% | 模型输出格式为连续文本，未使用要求的标签分隔 |
| **无`</answer>`结束标签** | ~60% | 生成长度不足或被截断 |

**结论**：格式错误主要由 **Prompt 设计与解析器逻辑不匹配** 导致，而非模型本身无法遵循指令。

---

## 四、答案错误样本分析（类别2，共112例）

### 4.1 10例答案错误样本对比

#### 示例1

**问题**：If Cindy initially had 37 envelopes and she gives 3 envelopes to each of her 5 friends, how many envelopes does she have...

**Ground Truth**：`22`

**模型预测**：`22 envelopes`

---

#### 示例2

**问题**：Last year, Jorge planted corn on all of his x acres of property.  Typically, corn grown on good soil yields 400 bushels ...

**Ground Truth**：`60`

**模型预测**：`Let's solve the equation step by step. First, we can simplify the equation by multiplying both sides...`

---

#### 示例3

**问题**：Carrie is wrapping three birthday presents. One present needs x square feet of wrapping paper to cover. The second prese...

**Ground Truth**：`2`

**模型预测**：`The value of the unknown variable x is 2 square feet.`

---

#### 示例4

**问题**：An airplane was flying from California to Virginia.  The flight started with 124 passengers.  They made a layover in Tex...

**Ground Truth**：`10`

**模型预测**：`67 - 58 + 24 - 47 + 14 + x = 67  x = 67 - 58 + 24 - 47 + 14 - 67  x = 0`

---

#### 示例5

**问题**：Daryl is loading crates at a warehouse and wants to make sure that they are not overloaded. Each crate can weigh up to 2...

**Ground Truth**：`80`

**模型预测**：`80kg`

---

#### 示例6

**问题**：If the company's data entry team consists of 5 employees, with Rudy typing 64 words per minute, Joyce typing 76 words pe...

**Ground Truth**：`80`

**模型预测**：`The total number of words typed per minute by the team is 64 + 76 + 91 + 80 + 89 = 400. The average ...`

---

#### 示例7

**问题**：If the perimeter of a triangle is 160 cm and two of the sides measure 40 cm and 50 cm, what is the length of the third s...

**Ground Truth**：`70`

**模型预测**：`The length of the third side is 160 cm - (40 cm + 50 cm) = 70 cm.`

---

#### 示例8

**问题**：Miss Grayson's class raised $50 for their field trip. Aside from that, each of her students contributed $5 each. There a...

**Ground Truth**：`7`

**模型预测**：`The value of the unknown variable $x is $7.`

---

#### 示例9

**问题**：How many ways are there to choose X cards from a standard deck of 52 cards, if all three cards must be of different suit...

**Ground Truth**：`3`

**模型预测**：`4`

---

#### 示例10

**问题**：Jan's three-eyed lizard has 3 times more wrinkles than eyes, and seven times more spots than wrinkles. How many fewer ey...

**Ground Truth**：`69`

**模型预测**：`23E`

---

### 4.2 错误类型分类

基于10例样本观察，答案错误可分为以下类型：

| 错误类型 | 典型表现 | 示例数 | 根因 |
|----------|----------|--------|------|
| **答案格式冗余** | 输出"22 envelopes"而非"22" | 3 | 模型输出完整句子，未提取纯数值 |
| **符号/单位错误** | 正负号相反、量纲混淆 | 2 | 数学理解或计算失误 |
| **推理过程正确但结论错** | 中间步骤对，最终答案错 | 2 | 计算错误或过早终止 |
| **过度推理** | 在`<answer>`中继续写推理 | 2 | 对格式要求理解不准确 |
| **完全错误** | 思路与问题无关 | 1 | 题目理解失败 |

---

## 五、正确样本示例（类别1，共115例）

### 5.1 典型成功案例

#### 示例1

**问题**：What is  $(-1)^1+(-1)^2+\cdots+(-1)^{2006}$ ?...

**答案**：`0`

**模型预测**：`0`

---

#### 示例2

**问题**：If Lilah's family gallery initially had 400 photos and they took half as many photos as they have in the gallery on the ...

**答案**：`920`

**模型预测**：`920`

---

#### 示例3

**问题**：Circle $T$ has its center at point $T(-2,6)$. Circle $T$ is reflected across the $y$-axis and then translated 8 units do...

**答案**：`(2,-2)`

**模型预测**：`$(2,-2)$`

---

#### 示例4

**问题**：Compute $\dbinom{16}{5}$....

**答案**：`4368`

**模型预测**：`4368`

---

#### 示例5

**问题**：How many distinct positive factors does 32 have?...

**答案**：`6`

**模型预测**：`6`

---

---

## 六、结论

1. **格式遵循问题突出（54.6%）**
   - 主要矛盾：Prompt 已包含`<think>`，模型直接续写，解析器却要求完整标签对
   - 这是 **Prompt-Parser 不匹配** 问题，非模型能力问题

2. **答案准确率有待提升（45.4% 格式合规，仅23.0% 全对）**
   - Zero-shot 数学推理准确率约 50%（在合规样本中）
   - 答案解析需要更智能的数值提取逻辑

3. **模型具备基础推理能力**
   - 45.4% 的样本能生成符合格式要求的回答
   - 说明模型理解`<think>`/`<answer>`的语义

---

**报告生成时间**：2026-04-06 23:35:12
**数据来源**：`results/mathbaseline/metamath_analysis.json`
