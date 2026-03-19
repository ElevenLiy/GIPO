# AdaMacro: Budgeted Skill Discovery by Consolidating Frequent Tool Trajectories

## 1. Introduction

大语言模型驱动的工具调用型 Agent 已成为自动化复杂任务的重要方式。尽管任务指令千差万别，实际交互轨迹显示，Agent 经常重复执行相似的短程子流程，例如"检索后选择目标并打开"、"抓取后解析并写入"或"读取文件后运行命令"。这些重复片段表明，若能将高频短程步骤固化为可复用单元，便有机会缩短有效决策链路，降低探索成本，并显著提升训练时的样本利用率。

近期经验重用方向主要依赖 Memory 与自我进化框架。常见做法是将历史交互片段检索作为上下文参考，或将经验总结为文字建议注入提示，从而辅助模型规划。与此同时，FaSTA* 等工作开始探索将复用单元进一步推进到动作层，尝试挖掘可执行的子程序。这一趋势表明，从"文本记忆的上下文层复用"走向"结构化的动作层复用"是一条提升 Agent 效率的自然路径。

然而，将复用推进到动作层并不等价于把经常相邻的工具名直接拼接成新动作。**第一**，基于局部共现的朴素拼接会导致动作空间膨胀。若缺乏明确的预算与筛选机制，近似变体会导致宏库规模迅速增长，反而增加了选择难度与冲突概率。**第二**，纯符号拼接难以提供一致的执行语义。宏动作不仅是顺序列表，还需具备"候选选择"、"异常处理"与"中断回退"能力；否则，一旦宏在高频使用中失败，不仅难以定位具体错误阶段，强化学习的训练信号也难以精确分配到宏内部。因此，核心难点在于如何在基数约束下，选出最具压缩收益的片段，并将其转化为既可执行又可诊断的技能。

为此，我们提出 **AdaMacro**。我们将宏发现建模为一个带基数约束的行为压缩问题：在给定预算下，从成功轨迹中利用 BPE 式迭代合并算法，优先挖掘高频且高收益的子流程，并将其转化为模板化的可执行技能（Executable Skills）注册到工具库中。为了解决执行与训练难题，AdaMacro 采用两项关键设计：其一，保留原子工具与宏技能并存，使 Agent 可灵活决策；其二，在运行时注入 Trace 追踪与软中断机制，使宏具备可定位、可回退的统一语义，并支持基于 SFT 与 GRPO/GIPO 的分阶段反馈训练。实验表明，该方法在有效控制动作空间规模的同时，显著提升了 Agent 的执行效率与鲁棒性。

**贡献总结：**

第一，我们提出带约束的 BPE 式宏挖掘，在成功轨迹上自动发现高频子流程，并用词表预算与轻量裁剪控制宏库规模。

第二，我们提出模板化的技能（skills）实例化与 trace 感知执行机制，将工具宏转化为语义明确的可执行技能，统一候选选择，异常处理，中断与回退，并与原子工具完全并存兼容。

第三，我们在 SFT + GRPO 框架下引入"宏整体奖励 + 效率奖励 + 宏内部阶段反馈"的组合回报设计，利用组内相对比较稳定学习混合动作空间中的宏选择与内部策略参数选择。进一步提出 **GIPO（Granularity-Imagination Policy Optimization）**，通过逐步反事实想象比较原子与技能粒度的覆盖差异，提供密集的过程级奖励信号。并在 TOOLATHLON 与 TOUCAN 上系统评估效率与性能收益。预计在 Qwen2.5-1.5b、Qwen2.5-7b、Llama3.1-8b、Llama3.2-3b 上进行实验。

---

## 2. Method

### 2.1 带预算约束的 BPE 宏挖掘

我们从训练日志中抽取成功轨迹，将每条轨迹映射为工具 token 序列。为减少具体参数值噪声并提升泛化能力，每个原子工具调用被抽象为"工具名 + 关键参数签名"的 token，例如 `Search[q]`、`Click[id]`。在该表示下，宏挖掘被建模为一个带预算的序列压缩过程。我们采用 BPE 式迭代合并：统计语料中相邻 token 对的出现次数，并在每轮合并中选择计数最大的相邻对进行合并。

给定轨迹语料 $\mathcal{D}$，相邻对 $(u, v)$ 的加权计数定义为：

$$C(u, v) = \sum_{\tau \in \mathcal{D}} \sum_{t=1}^{|\tau|-1} \mathbf{1}[x_t = u, x_{t+1} = v],$$

其中 $w(\tau)$ 默认对成功轨迹取 1。每轮选择出现最频繁的相邻对并合并为新 token：

$$(u^*, v^*) = \underset{(u,v)}{\operatorname{argmax}} C(u, v).$$

**预算约束**：最大合并轮数 $K=50$，最小频率阈值 `min_freq=3`，最大宏长度 `max_macro_len=6`（最多包含6个原子工具），最小宏长度 `min_macro_len=2`。合并完成后，使用率低于 `min_usage_ratio=0.01` 或频率低于阈值的宏被裁剪。宏获得语义化命名（如 `list_dir_and_read_file`），而非不透明的数字编号。

在理论层面，BPE 可以视作"在给定合并轮数预算下最大化压缩收益"的贪心算法。相关论文的分析表明，BPE 背后的最优问题具有近似难度，但贪心 BPE 依然能对压缩收益提供常数因子近似保证。在我们的场景中，工具序列同样是离散符号串，因此这一保证可作为选择 BPE 的理论支撑。我们需要补充一个对照实验，将 BPE 与"频繁 n-gram 挖掘并选 top-K 宏"的方法直接比较，突出 BPE 在预算下得到更紧凑、冗余更低的宏库这一点。

### 2.2 自动技能实例化与 trace 感知

BPE 产出的是符号层面的宏 token。为使 LLM 能像选择普通工具一样选择宏，我们将每个宏 token $m$ 实例化为可执行技能（skills）$S_m$，并注册到工具库中，同时保留全部原子工具。技能实例化采用模板化方式实现。模板集合覆盖常见结构：

- **sequential**：纯顺序执行模板
- **select**：上一步输出候选列表，下一步需要从候选中选取输入的选择模板
- **conditional**：基于输出条件分支的模板

模板类型通过 `SkillTemplate.detect_template()` 自动检测：若某工具返回列表/数组类输出（名称含 search、list、query、find 等关键词），且下一步工具需要 id/index 等选择参数，则判定为 select 模板；否则为 sequential 模板。该设计避免直接生成任意代码带来的不确定性，并保证技能执行语义一致。

**参数自动确定**：基于工具 schema 与相邻步骤字段匹配，技能对外暴露的参数（exposed params）和内部自动传递的参数（internal params）由 `analyze_skill_parameters()` 自动分析确定：
- 第一个工具的必填参数始终暴露给 LLM
- 后续工具的参数若名称匹配前序工具输出字段，则标记为内部自动管道传递
- 其余未匹配参数仍暴露给 LLM

为保留有限的内部可控性，我们额外暴露离散化 `select_strategy` 参数（例如 rank-0、rank-1、random、filter），用于控制候选选择方式。也就是说为了不让宏变得太死板，我们在自动传递数据的同时，给 LLM 留了一个外部参数，让它能指挥宏在内部怎么挑数据。技能内部的中间变量由解释器自动传递，不要求模型显式填写。

**运行时 trace 与软中断**：我们对技能执行注入 trace。技能解释器 `SkillInterpreter` 按序展开内部原子工具调用，并记录每步工具名与状态码：

$$\text{Trace}(S_m) = [(tool_1, status_1), \ldots, (tool_k, status_k)].$$

若执行出现硬失败（超时、报错）或逻辑阻断（候选为空、断言失败），解释器触发**软中断**，停止后续展开并返回包含 trace 与中间观测的中断结果，将控制权交还给策略。由于原子工具始终保留，策略可退回原子动作逐步完成任务，或改选其他技能继续推进。

**增强工具库**：最终通过 `build_augmented_tools()` 构建增强工具库，将技能作为工具注册（带有正式 JSON schema），与原子工具并存：

$$\mathcal{A} = \mathcal{A}_{\text{atom}} \cup \mathcal{A}_{\text{skill}}.$$

### 2.3 SFT + GRPO 训练与分阶段奖励

在完成技能实例化后，训练的关键变成如何在"技能与原子并存"的动作集合里稳定学会选择。AdaMacro 采用两阶段训练流程。

#### 2.3.1 SFT 阶段

首先用 SFT 学习基本工具调用格式，并在工具列表中显式提供技能，使模型从一开始就把技能当作正常候选。SFT 数据从成功轨迹生成，每条轨迹产生 **4 种训练变体**：

1. **Atomic（原子变体）**：保留原始工具序列不变，教会模型正确的工具调用顺序
2. **Full skill（完全技能替换）**：将轨迹中可匹配的工具子序列全部替换为技能调用，教会模型技能的使用方式
3. **Partial skill（部分技能替换）**：仅替换第一个匹配的技能，其余保持原子展开，教会模型混合使用技能与原子工具
4. **Continuation（中间状态续写）**：给定已执行的前缀工具序列，预测后续剩余序列，教会模型从中间状态恢复与续写

训练使用 **LoRA**（rank=64, alpha=128, dropout=0.1），目标模块包括注意力层（q/k/v/o_proj）与 MLP 层（gate/up/down_proj），采用 **assistant-only loss masking**——仅对 assistant tokens（工具调用 + 最终总结）计算交叉熵损失。

SFT 超参数：epochs=3, lr=1e-5, batch_size=2, grad_accum=8, max_seq_len=4096。

#### 2.3.2 GRPO 阶段

随后进入强化学习阶段，采用 GRPO 做策略优化。AdaMacro 在混合动作空间中训练策略，动作集合由原子工具与技能并存构成。

选择 GRPO 的关键原因在于混合动作空间会显著增加"同一状态下的可行轨迹数量"。面对同一提示与相同上下文，通常同时存在用技能快速推进的有效轨迹与用原子工具逐步完成的有效轨迹。此时最直接、最稳定的学习信号来自这些可行轨迹之间的相对优劣比较。GRPO 对同一提示进行组采样，并在组内构造相对优势来更新策略，从而更适合该决策环境。具体目标为：

$$\max_\theta \mathbb{E}_p \left[ \frac{1}{G} \sum_{i=1}^{G} A_i \log \pi_\theta(y_i \mid p) - \beta \text{KL}(\pi_\theta(\cdot \mid p) \| \pi_{\text{ref}}(\cdot \mid p)) \right], \quad A_i = \text{normalize}(R_i).$$

**多样化 Rollout 策略**：对每个训练提示生成 $G=4$ 个 rollout，采用不同策略以增加探索多样性：

| Rollout | 策略 | 说明 |
|---------|------|------|
| g=0 | 低温利用 | 低温度采样，倾向于已学习的高概率动作 |
| g=1 | 正常温度 | 基础温度 (temperature=0.7) 的标准采样 |
| g=2 | 高温探索 | 1.8× 基础温度，鼓励探索未知动作 |
| g=3 | Oracle-seeded | 将一个 ground-truth 工具强制作为第一个动作，打破坍塌 |

**组合奖励函数**：回报由任务覆盖、技能奖励、效率奖励三部分共同构成：

$$R(\tau) = R_{\text{task}}(\tau) + R_{\text{skill}}(\tau) + R_{\text{efficiency}}(\tau)$$

**$R_{\text{task}}$（任务覆盖奖励，0-1 范围）**：衡量 rollout 中使用的工具对 ground-truth 工具的覆盖程度。采用三级模糊匹配：
- 精确匹配：工具名完全一致，得分 1.0
- 子串匹配：一方为另一方的子串，得分 0.8
- Jaccard 匹配：工具名分词后 Jaccard 相似度 ≥ 0.5，得分为 Jaccard 值

此外加入**位置感知的顺序奖励**：基于最长递增子序列（LIS）计算工具使用顺序与 GT 顺序的一致性，给予额外 order bonus。

**$R_{\text{skill}}$（技能奖励，-0.15 到 +0.1 范围）**：加法式的技能使用反馈信号：
- 技能链与 GT 工具有覆盖且全部成功完成：奖励 $0.05 + 0.05 \times \text{chain\_coverage}$
- 技能链与 GT 工具有覆盖但部分中断：按成功步骤比例折算
- 技能链与 GT 工具零覆盖（不相关技能调用）：惩罚 -0.05
- 重复调用同一技能：每次额外惩罚 -0.03
- 总技能奖励裁剪到 $[-0.15, +0.1]$，防止堆叠

该设计使训练信号携带失败位置与失败类型信息，并结合软中断机制保证策略可在技能失败后退回原子工具继续探索。

**$R_{\text{efficiency}}$（效率奖励）**：对决策步数给予效率奖励，但需满足**门控条件**（$R_{\text{task}} \geq 0.4$）才生效：
- 步数效率奖励：$\max(0, 1 - \frac{\text{steps}}{\text{max\_steps}}) \times 0.15 \times R_{\text{task}}$
- 压缩奖励：仅在技能实际使用且决策步数少于 GT 步数时授予，$(\text{gt\_steps} - \text{decision\_steps}) \times 0.05$
- **过短探索惩罚**：当决策步数 < 3 且 $R_{\text{task}} < 0.3$ 时，施加惩罚 $-0.05 \times (3 - \text{steps})$

总奖励裁剪为非负值（$\max(R, 0)$），避免负奖励导致 GRPO 优势估计不稳定。

**反坍塌机制**：在线 GRPO 训练中引入三项机制防止策略坍塌：
1. **Oracle seeding**：G=4 中的一个 rollout 使用 GT 工具作为首个动作
2. **Continuation nudge**：若模型过早停止（输出文本而非工具调用），强制模型继续调用工具
3. **0-step resample**：若所有 rollout 均产生 0 个动作，以更高温度重试

**策略梯度**：优势 = 组内标准化奖励。损失 = 优势 × 仅 assistant tokens 的交叉熵。梯度在 `grad_accum` 个提示上累积后执行一次优化器更新。

GRPO 超参数：epochs=3, lr=5e-6, group_size=4, kl_coeff=0.05, max_gen_length=512, temperature=0.7。

### 2.4 GIPO：粒度想象策略优化（Granularity-Imagination Policy Optimization）

GIPO 是 GRPO 的扩展，核心创新在于引入**逐步反事实想象（per-step counterfactual imagination）**，为策略提供密集的过程级奖励信号。

**核心机制**：在每个工具调用步骤，若模型选择了某个粒度的动作（原子或技能），GIPO 检查是否存在另一粒度的替代动作，并模拟其执行来比较 GT 工具覆盖差异：

- 模型调用原子工具 $X$ → 检查是否存在包含 $X$ 的技能 → 模拟技能执行 → 比较覆盖 → 生成该步的过程奖励
- 模型调用技能 $S$ → 提取 $S$ 链中的第一个原子工具 → 模拟原子执行 → 比较覆盖 → 生成该步的过程奖励

**训练流程**：

1. **Phase 1**：生成 2 个基础 rollout（base_0 使用技能偏置提示，base_1 使用中性提示）
2. **Phase 1.5**：在每个工具调用步骤，若存在替代粒度，分叉反事实分支。分支数为 0-2，总组大小动态变化（2-4 个 rollout）
3. **Phase 2**：计算组合奖励

$$R = R_{\text{task}} + R_{\text{skill}} + R_{\text{efficiency}} + R_{\text{imagination}}$$

其中 $R_{\text{imagination}}$ 是来自反事实粒度比较的过程奖励：
- 每步想象奖励裁剪：`gipo_step_reward_cap = 0.1`
- 每 rollout 总想象奖励裁剪：`gipo_total_reward_cap = 0.3`
- 想象奖励缩放：`gipo_step_reward_scale = 0.15`

**参数映射**：反事实动作通过参数映射（parameter mapping）接收有意义的参数值，确保模拟结果可靠。

**GIPO-API 变体**：将静态 `tool_simulator_database.json` 替换为外部 LLM API（如 DeepSeek V3.2 / Qwen3.5-plus via DashScope）进行工具输出模拟，获得更真实的训练信号。

---

## 3. Experiment

### 3.1 数据集与设置

我们在 TOOLATHLON 与 TOUCAN 上进行评估。宏挖掘仅使用训练集成功轨迹构建技能库，评估阶段冻结技能库以保证可复现。为保证公平比较，我们统一 backbone、工具接口与训练预算，并以真实原子工具调用次数作为主要预算约束标准，避免技能降低决策次数带来的不公平优势。

训练/测试划分采用 70/30 分割，使用后 30% 的 episodes 作为测试集。

实验在以下模型上进行：
- **Qwen2.5-1.5B-Instruct**
- **Qwen2.5-7B-Instruct**
- **Llama3.1-8B-Instruct**
- **Llama3.2-3B-Instruct**

训练方法包含三个 Step 4 变体：
1. **AdaMacro-GRPO**：基础 GRPO 训练（结果级奖励）
2. **AdaMacro-GIPO**：GIPO 训练（结果级 + 过程级反事实想象奖励）
3. **AdaMacro-GIPO-API**：使用外部 LLM API 模拟工具输出的 GIPO 变体

### 3.2 对比方法

我们按分类的形式组织基线：仅原子工具的 RL/SFT-RL 方法；基于检索的 raw-context memory；基于反思的显式指导 memory；自我进化式记忆维护方法；以及具备可执行技能或程序化复用能力的相关框架（近期应该会有）。

### 3.3 评估指标

**主指标：**
- **Pass Rate**：任务成功率（GT 工具覆盖 ≥ 50% 的 episodes 比例）
- **Pass Rate Strict**：严格成功率（GT 工具覆盖 = 100% 的 episodes 比例）
- **Avg Tool Calls**：平均决策步数（技能调用算 1 步）
- **Coverage**：平均 GT 工具覆盖率，含三级模糊匹配 + 顺序奖励

**机制指标：**
- 技能使用比例（Skill Usage Ratio）
- 技能覆盖率（Skill Coverage）
- 技能成功率（Skill Success Rate）
- 技能中断比例与中断位置分布（Skill Interrupt Ratio & Position Distribution）
- 不同 select_strategy 的命中率与贡献

### 3.4 消融实验

我们进行以下消融：
- 去掉 BPE 预算控制或裁剪（对比无约束 n-gram 挖掘）
- 去掉模板化实例化仅保留符号宏
- 去掉 trace 与软中断
- 只保留 R_task，不使用 R_skill 和 R_efficiency
- R_efficiency 的门控阈值敏感性（0.3 / 0.4 / 0.5）
- 技能库规模（max_merges = 20 / 50 / 100）与技能最大长度（max_macro_len = 4 / 6 / 8）的敏感性分析
- GRPO vs GIPO vs GIPO-API 的对比
- SFT 数据变体的贡献消融（去掉 continuation / 去掉 partial_skill）
- 反坍塌机制消融（去掉 oracle seeding / continuation nudge / 0-step resample）
