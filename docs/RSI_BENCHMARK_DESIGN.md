# RSI benchmark 设计稿

2026-11-03 起草，同日确认并开始实现（实现状态见第 10 节）。本文是接口与实验设计，不是结果；
Beacon 尚未运行。
目标论文：ACL 2027（ARR 截止 2027-01-04），资源类长文。备选：NeurIPS 2027 Datasets and
Benchmarks track。

论文标题（2026-11-03 确定）：**Fin-RSI: An Agent-Native Library of Financial Skills, Checks,
and Self-Improvement**。RSI 是库的一部分；标题不提 benchmark，不预设实验结论。

## 1. 要回答的问题

在同一个模型、同一个裁判、同样预算下，**哪种自我改进（RSI）结构能让 agent 在没见过的
金融编程题上写出更多有效程序**；加上 fin-skills 库之后，这个结论是否改变。

主要终点：留出题上的**有效交付率**（被裁判接受，且隐藏评分器判为正确）。
次要终点：接受但错误率、模型调用次数、生成 token、墙钟时间、库调用率。

不回答：收益率、实盘表现、模型能力的一般排名。交易实验在论文里作为案例单列。

## 2. 统一接口：`fin_skills.rsi.methods`

所有方法继承同一个基类，用法与 PyOD 的 `fit` / `decision_function` 对应：

```python
from fin_skills import rsi
from fin_skills.tasks import suite

tasks = suite()                                   # 冻结的题库，含 dev / heldout 划分
budget = rsi.Budget(attempts=3, calls=6, generated_tokens=12_000)

method = rsi.Reflexion(library=True)
method.fit(generate, tasks.dev, budget=rsi.FitBudget(episodes=48, proposals=6))
episode = method.solve(generate, tasks.heldout[0], seed=11, budget=budget)
episode.accepted, episode.code, episode.usage      # calls, generated_tokens, attempts, roles
# episode.valid is filled in only by rsi.benchmark(), from the hidden grader

report = rsi.benchmark(rsi.METHODS, generate, tasks, seeds=(11, 23, 37),
                       library=(False, True), budget=budget)
```

### 2.1 三个对象

| 对象 | 内容 | 现有代码 |
|---|---|---|
| `generate(messages) -> str` | 用户自己的模型，消息格式与 `agents.Team` 相同 | `agents/team.py` |
| `Task` | id、family、split、traits、题面、`inputs(seed)`、裁判检查列表、隐藏评分器 | 新增 `fin_skills/tasks/` |
| 裁判 `judge(code, task, data) -> Attempt` | 在沙箱里执行，跑检查，返回 `rsi.wiring.Attempt` | `codegen_controller_worker_v2.py`、`linux_sandbox.py` |

隐藏评分器只产生分数，**永远不进入任何反馈或提议上下文**（沿用 `rsi.wiring` 的约束）。

### 2.2 基类

```python
class BaseRSI:
    name: str
    def __init__(self, *, library: bool = False): ...
    def fit(self, generate, dev_tasks, *, budget: FitBudget) -> "BaseRSI": ...  # 默认不做事
    def solve(self, generate, task, *, seed: int, budget: Budget) -> Episode: ...
    def state(self) -> dict: ...          # 学到了什么（经验、反思、手册），可哈希、可审计
```

- `library=False`：题面不给库接口说明，规则卡用 `tools=False` 渲染，沙箱禁止
  `import fin_skills`。
- `library=True`：接口说明**从函数签名自动生成**（修正 v2 里 `data_dictionary=` 与真实参数
  `data` 不一致的问题），规则卡带库调用提示，库调用被追踪。
- 方法与库开关正交，所以每种方法都有"有库 / 无库"两组。

### 2.3 公平预算（由计量器强制，不靠自觉）

所有模型调用都经过 `rsi.Meter`。超出预算的调用直接拒绝，并记入记录：

| 预算 | 含义 | 默认 |
|---|---|---|
| `Budget.attempts` | 每题最多执行几次程序 | 3（与 v1/v2 相同） |
| `Budget.calls` | 每题最多几次模型调用，**含反思、审查者等辅助调用** | 6 |
| `Budget.generated_tokens` | 每题生成 token 上限 | 12,000 |
| `FitBudget.episodes` | 开发阶段最多做几道开发题 | 48 |
| `FitBudget.proposals` | 开发阶段最多几次经验 / 手册修改提议 | 6 |

不学习的方法（NoImprove、SelfDebug、CheckFeedback）`fit` 为空，但报告里照样列出它们的
开发阶段花费（0），这样"学习花了多少额外算力"可以直接比较。

## 3. 方法清单

| 方法 | 测试时怎么做 | 开发阶段学什么 | 复用的代码 | 文献中的对应 |
|---|---|---|---|---|
| `NoImprove` | 生成一次 | 无 | E4 单次生成 | 基线 |
| `SelfDebug` | 失败后只回传报错，重试 | 无 | v1/v2 `raw_exec` 组 | Self-Debugging |
| `CheckFeedback` | 失败后回传失败的检查、原因和规则卡 | 无 | v2 `decide()` + `rules.explain()` | 带验证器的反馈 |
| `Reflexion` | 失败后先让模型写一段反思，再带反思重试 | 可选：跨题保留反思 | 新增 | Reflexion |
| `PlaybookLearning` | 把冻结的经验手册放进提示，再按 CheckFeedback 重试 | 在开发题上用 `RSILoop` + `no_regression` 积累手册 | `rsi.core`、`rsi.wiring` | ExpeL、Dynamic Cheatsheet |
| `Team` | 编写者 + 按规则族的审查者 | 无 | `agents.Team` | 多角色审查 |
| `Evolve` | 用开发阶段进化出的提示词 | 在开发题上迭代改写系统提示 | 新增 | GEPA 类提示进化 |

2026-11-03 决定：7 种全部纳入，包括 `Evolve`。`rsi.core` 里已有数据类 `Playbook`，所以方法类名为
`PlaybookLearning`（`name = "playbook"`）。

## 4. 题库：`fin_skills.tasks`

### 4.1 来源

从算法目录里**不依赖可选库、状态为 ready** 的算法生成题目，每道题都有一个**不调用
fin-skills 的独立参考实现**（与 `codegen_utility.reference()` 同样的写法）。现有可用：

| 类型 | 算法 | 数量 |
|---|---|---:|
| portfolio | equal_weight, inverse_volatility, min_variance, hrp, cross_sectional_momentum | 5 |
| signal | ma_crossover, momentum, vol_target_momentum, macd, bollinger_reversion, donchian_breakout, rsi_reversion | 7 |
| risk | historical_var_es, normal_var_es | 2 |
| volatility | historical_volatility, ewma_volatility | 2 |
| forecast | naive, drift, mean, seasonal_naive | 4 |
| execution | twap, vwap | 2 |
| pricing | black_scholes, american_crr | 2 |

仓位类（portfolio + signal，12 个算法）每个做 2 个变体，共 24 道。其余类型每个算法 1 道，
共 12 道。合计约 36 道。

### 4.2 每道题内置的"坑"

坑来自实际观察到的失败，每道题声明自己的 traits，自动选规则卡：

- 拆股因子：累积因子 vs 逐次比率（变体之一换成 `factors="ratio"`）
- 执行时滞：当根收盘价不能决定当根仓位
- 预热期：窗口不满时必须为 0、不能留 NaN
- 首行收益：`pct_change()` 的首行缺失
- 列顺序：变体之一打乱资产列顺序，检验对齐
- 年化与单位（risk、volatility、pricing）

### 4.3 开发 / 留出划分

- **按算法划分**：同一个算法的所有变体只出现在一边，留出题的算法在开发阶段从未出现。
- E4 的 4 道题（diagonal_risk_parity、cross_sectional_momentum、ma_crossover、
  vol_target_momentum）在 v1/v2 迭代中已被用来设计检查和修库，**只能放在开发集**。
- 其余算法按类型分层，用固定种子抽签，约 1/3 进开发集、2/3 进留出集。抽签结果写进
  `protocol.json` 后冻结。
- 每道题 3 个数据种子（11、23、37），种子不算独立题目，统计时按题聚类。

### 4.4 题目资格检查（推理前必须全过）

1. 参考实现与库函数在每个种子上一致（误差 ≤ 1e-8）。
2. 参考实现通过该题全部裁判检查。
3. 每类坑都有一个"植入错误"版本，必须被对应检查拒绝（沿用 v2 qualification 的做法）。
4. 隐藏评分器与裁判检查是两套代码，评分器不出现在任何提示里。

## 5. 统计与汇报

- 每个模型、每种方法、有库 / 无库，报告留出题有效交付率及按题聚类的 bootstrap 95% 区间。
- 主要比较：各方法对 `NoImprove`，以及有库对无库，配对到同一题同一种子；多重比较用
  Holm 校正。
- 同时报告每个结果的调用次数、生成 token、墙钟时间，和开发阶段的额外花费。
- 分析方案在推理前写进 `protocol.json`。结果不管方向如何都完整报告；不重跑、不挑种子、
  没有 v4。

## 6. 实验矩阵与算力

模型：从 Beacon 已缓存、无需下载的 13 个模型中选 5 个，覆盖至少 4 个家族。资格检查要求
上下文 ≥ 16k token（v1 的平均提示已有约 11k token），并能在单卡上按原精度运行。
Qwen2.5-Coder 7B/14B 参与过开发，若入选需在论文中标为开发模型。

| 维度 | 数量 |
|---|---:|
| 方法 | 7 |
| 库开关 | 2 |
| 留出题 | 16 |
| 种子 | 3 |
| 模型 | 5 |
| 合计 episode | 约 3,400，另加开发阶段 |

粗略算力：v2 在 L40S 上 72 个 episode 用了 1 小时 25 分，平均约 71 秒一个。按此估计约
70 GPU 小时，反思、审查类方法调用更多，实际会更高。这是估计，不是承诺；冻结协议前用
每个模型 2 道开发题实测再定。

### 6.2 组件消融：技能、检查、自我改进各自起多大作用（2026-11-04 设计，未运行）

标题里的三个部分分开测。技能有两部分：规则卡（文字知识）和库调用（可执行工具），用两个
开关组合成四组（`rsi.ARMS`）：

| 组 | 系统提示里的规则卡 | 库接口，可 `import fin_skills` | 反馈里的规则卡 |
|---|---|---|---|
| `none` | 无 | 否 | 无，只有失败的检查和原因 |
| `cards` | 有（不含库调用） | 否 | 有 |
| `library` | 无 | 是 | 无，只有失败的检查和原因 |
| `cards+library` | 有 | 是 | 有 |

- 四组都保留题目本身的定义：规则、时滞说明、复权因子的形式。这些规定了正确答案，不算技能。
- 检查失败时，四组都看到检查名和原因（检查观察到了什么，例如"只改动当日及以后的价格，
  仓位却变了"）。原因属于"检查"这部分；"该怎么改"的规则卡属于"技能"。
- 方法：`NoImprove`（不看反馈）、`CheckFeedback`（看检查反馈）、`PlaybookLearning`（在开发
  题上学经验）。同一组内，NoImprove→CheckFeedback 的差别是检查的作用，
  CheckFeedback→PlaybookLearning 的差别是自我改进的作用；同一方法跨组的差别是技能的作用。
- `cards` 和 `cards+library` 两组就是主实验的无库组和有库组，可直接复用；只需新增 `none`
  和 `library` 两组：3 种方法 × 2 组 × 16 道留出题 × 3 个种子 × 5 个模型 = 1,440 个
  episode。按 71 秒一个粗估约 28 GPU 小时，这是估计，冻结协议前按实测再定。
- 主要比较配对到同一题、同一种子、同一模型，与第 5 节一样用按题聚类的 bootstrap 和 Holm
  校正。

### 6.1 模型元数据资格检查（2026-11-03，只读，未加载权重）

通过已有的共享 SSH 连接（未新建登录）读取 10/17 运行留下的缓存清单，以及每个已缓存模型在
固定 revision 下的 `config.json` / `tokenizer_config.json`。20 个清单条目中 13 个已缓存，
7 个未缓存（不下载）。存储剩余 216 GB（96% 已用）。

| 模型 | 家族 | 上下文 | 权重 (GB, bf16) | 单卡 | 备注 |
|---|---|---:|---:|---|---|
| qwen-coder-7b | Qwen | 32,768 | 15.2 | 48 GB 卡可 | 开发模型（v1/v2） |
| qwen-coder-14b | Qwen | 32,768 | 29.5 | 48 GB 卡可 | 开发模型（v1/v2） |
| qwen-coder-32b | Qwen | 32,768 | 65.5 | 需 80 GB 级 | |
| qwen-general-7b | Qwen | 32,768 | 15.2 | 48 GB 卡可 | |
| qwen3-4b | Qwen | 262,144 | 8.0 | 48 GB 卡可 | |
| mistral-nemo-12b | Mistral | 131,072 | 24.5 | 48 GB 卡可 | |
| mistral-small-24b | Mistral | 32,768 | 47.1 | 需 80 GB 级 | 48 GB 卡放不下权重加 KV |
| phi4-mini | Phi | 131,072 | 7.7 | 48 GB 卡可 | |
| granite-8b | Granite | 131,072 | 16.3 | 48 GB 卡可 | v11 交易研究预检曾因输出格式失败 |
| olmo3-7b | OLMo | 65,536 | 14.6 | 48 GB 卡可 | |
| falcon3-10b | Falcon | 32,768 | 20.6 | 48 GB 卡可 | |
| smollm3-3b | SmolLM | 65,536 | 6.2 | 48 GB 卡可 | |
| deepseek-r1-distill-14b | Qwen 架构 | 131,072 | 29.5 | 48 GB 卡可 | 推理模型；单次 2,048 token 上限会截断思考过程 |

全部为 bf16，带 chat template，上下文都 ≥ 32k，满足"≥ 16k"的要求。这一步不能代替真实
加载与生成；GPU 资格检查（加载、短生成、输出能解析出程序）在 Beacon harness 写好后做。

**候选 5 个**（5 个家族，3.8B–32B，均未参与开发）：qwen-coder-32b、mistral-small-24b、
granite-8b、olmo3-7b、phi4-mini。**替补**：falcon3-10b、mistral-nemo-12b、qwen3-4b。
不选 deepseek-r1-distill-14b 作主力：固定的单次生成上限会让它的结果主要反映截断，而不是
RSI 方法的差别。Qwen2.5-Coder 7B/14B 可以额外作为开发参照报告。

执行纪律（仓库 `AGENTS.md`）：fin-skills 同时最多 1 个 Slurm 作业（含排队），按模型串行
提交；只用 CPU 做资格检查；修库期间不扩大模型实验。2026-11-03 用户确认 GPU 小时不设上限；
单作业规则不变。

## 7. 要新增的代码

| 位置 | 内容 |
|---|---|
| `fin_skills/tasks/` | 题目定义、独立参考实现、数据生成、划分；加入 `HAND_WRITTEN_DIRS` |
| `fin_skills/rsi/methods.py` | `BaseRSI` 与七种方法 |
| `fin_skills/rsi/meter.py` | 调用与 token 计量、预算强制 |
| `fin_skills/rsi/bench.py` | `benchmark()` 运行器与汇报 |
| `benchmarks/agent_study/rsi_bench.py` | Beacon 端：冻结、资格检查、推理、评分，复用 v2 沙箱与 worker |
| `tests/` | 每种方法用脚本化的假模型测试；预算强制；划分不相交；隐藏评分器不泄漏；参考实现对库函数 |

## 8. 时间表

| 时间 | 内容 |
|---|---|
| 11/4–11/16 | 基类、计量器、七种方法；接口说明自动生成（11/3 已提前完成） |
| 11/17–11/27 | 题库、参考实现、资格检查；选模型；冻结协议 |
| 11/28–12/10 | Beacon 串行跑完整矩阵 |
| 12/1 | 接口冻结 |
| 12/10–12/28 | 写论文、合作者内审 |
| 12/29–1/4 | 匿名代码包、提交 |

## 9. 已确认的决定（2026-11-03）

1. 方法：7 种全部纳入（含 `Evolve`）。
2. 模型：先做资格检查，再给出 5 个候选。
3. Beacon 算力：GPU 小时不设上限；仍遵守 fin-skills 单作业、串行提交的规则。

## 10. 实现状态（2026-11-03）

已实现并测试（CPU，脚本化假模型，未调用真实 LLM）：

| 部分 | 位置 | 状态 |
|---|---|---|
| 计量器与预算 | `fin_skills/rsi/meter.py` | 完成；可嵌套（开发阶段的 episode 同时计入 fit 预算） |
| 7 种方法 | `fin_skills/rsi/methods.py` | 完成；`PlaybookLearning` 与 `Evolve` 在能改进时接受、不能改进时拒绝，均有测试 |
| 运行器与汇报 | `fin_skills/rsi/bench.py` | 完成；逐条写 JSONL，按题聚类 bootstrap |
| 题目、数据、参考实现 | `fin_skills/tasks/` | 34 道题（17 个算法 × 2 个变体）；全部 34 × 3 种子与库函数一致（见下） |
| 裁判与隐藏评分器 | `fin_skills/tasks/judge.py`、`worker.py` | 完成；4 类植入错误各被对应检查拒绝 |
| 库接口说明 | `fin_skills/tasks/prompts.py` | 从函数签名与 docstring 生成；默认参数有测试 |
| `agents.Team` | `tools` 参数 | 新增；无库组不再泄漏库调用提示 |
| 技能开关 | `cards` 参数（方法、`messages()`、`agents.Team`、`wiring.traces()`）；`rsi.ARMS`；`benchmark(arms=...)` | 2026-11-04 新增，见 6.2 |

题库现为 **34 道**（2026-11-03 扩充），仍为**临时**，冻结协议时最终确认：

| 类型 | 算法 | 题数 | 库里的一步调用 |
|---|---|---:|---|
| portfolio | inverse_volatility, cross_sectional_momentum, min_variance, hrp | 8 | `positions()` |
| signal | ma_crossover, vol_target_momentum, momentum, macd, bollinger_reversion, rsi_reversion, donchian_breakout | 14 | `positions()` |
| volatility | historical_volatility, ewma_volatility | 4 | `rolling()`（新增） |
| risk | historical_var_es, normal_var_es | 4 | `rolling()` |
| forecast | drift, mean | 4 | `rolling()` |

- 34 道 × 3 个种子的独立参考实现全部与库一致（min_variance 在 2e-6 内，其余 ≤ 3e-14）。
- 不做题的：equal_weight（输出与价格无关，坑测不出来）；naive 预测（等于平移一行）；
  seasonal_naive、execution、pricing（不适合价格面板的题型）。
- 划分：开发集 18 道（9 个算法），留出集 16 道（8 个算法），留出集覆盖全部 5 个类型。
  E4 的 4 个算法和资格检查作业用过的 momentum 强制放开发集：推理前被模型跑过的算法都算
  开发数据。
- 新增 `fin_skills.algorithms.rolling()`：波动率、风险、预测类算法按日滚动、滞后一根、
  复权一次，输出日期 × 资产表格。
- 扩题时发现并修复一个库缺陷：long-only `min_variance()` 在日收益协方差上 SLSQP 提前停止，
  权重与精确解（枚举子集）最多差 1.2e-2；现在在单位尺度上求解，误差 ≤ 1.4e-6。

- 加技能开关时发现并修复一处泄漏：`PlaybookLearning` 交给提议者的失败记录调用
  `rules.explain()` 时总带库调用提示，无库组学到的 playbook 可能叫模型调用不可用的
  fin_skills。现在 `wiring.traces()` / `observe()` 按本组的 `tools` 和 `rules` 生成，有测试。
  此前没有任何真实模型跑过 PlaybookLearning，不影响已有结果。

下一步：主实验 harness 的 run 阶段（含 6.2 的两组）；资格检查结果出来后冻结协议。
