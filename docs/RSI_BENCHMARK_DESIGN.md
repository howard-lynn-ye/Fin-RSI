# RSI benchmark 设计稿（待确认，未实现）

2026-11-03 起草。本文是接口与实验设计，不是结果；代码、题库和 Beacon 运行都尚未开始。
目标论文：ACL 2027（ARR 截止 2027-01-04），资源类长文。备选：NeurIPS 2027 Datasets and
Benchmarks track。

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
episode.valid, episode.accepted, episode.calls, episode.tokens, episode.attempts

report = rsi.benchmark([rsi.NoImprove, rsi.SelfDebug, rsi.CheckFeedback, rsi.Reflexion,
                        rsi.Playbook, rsi.Team], generate, tasks, seeds=(11, 23, 37),
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
| `Playbook` | 把冻结的经验手册放进提示，再按 CheckFeedback 重试 | 在开发题上用 `RSILoop` + `no_regression` 积累手册 | `rsi.core`、`rsi.wiring` | ExpeL、Dynamic Cheatsheet |
| `Team` | 编写者 + 按规则族的审查者 | 无（可接 `Playbook`） | `agents.Team` | 多角色审查 |
| `Evolve`（可选） | 用开发阶段进化出的提示词 | 在开发题上迭代改写系统提示 | 新增 | GEPA 类提示进化 |

`Evolve` 是否纳入取决于时间，11/16 前决定；不纳入就在论文里写明。

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
| 方法 | 6（不含 Evolve） |
| 库开关 | 2 |
| 留出题 | 约 24 |
| 种子 | 3 |
| 模型 | 5 |
| 合计 episode | 约 4,300，另加开发阶段 |

粗略算力：v2 在 L40S 上 72 个 episode 用了 1 小时 25 分，平均约 71 秒一个。按此估计约
85 GPU 小时，反思、审查类方法调用更多，实际会更高。这是估计，不是承诺；冻结协议前用
每个模型 2 道开发题实测再定。

执行纪律（仓库 `AGENTS.md`）：fin-skills 同时最多 1 个 Slurm 作业（含排队），按模型串行
提交；只用 CPU 做资格检查；修库期间不扩大模型实验。超出算力时的降级方案：3 个模型、
2 个种子。

## 7. 要新增的代码

| 位置 | 内容 |
|---|---|
| `fin_skills/tasks/` | 题目定义、独立参考实现、数据生成、划分；加入 `HAND_WRITTEN_DIRS` |
| `fin_skills/rsi/methods.py` | `BaseRSI` 与六种方法 |
| `fin_skills/rsi/meter.py` | 调用与 token 计量、预算强制 |
| `fin_skills/rsi/bench.py` | `benchmark()` 运行器与汇报 |
| `benchmarks/agent_study/rsi_bench.py` | Beacon 端：冻结、资格检查、推理、评分，复用 v2 沙箱与 worker |
| `tests/` | 每种方法用脚本化的假模型测试；预算强制；划分不相交；隐藏评分器不泄漏；参考实现对库函数 |

## 8. 时间表

| 时间 | 内容 |
|---|---|
| 11/4–11/16 | 基类、计量器、六种方法；接口说明自动生成 |
| 11/17–11/27 | 题库、参考实现、资格检查；选模型；冻结协议 |
| 11/28–12/10 | Beacon 串行跑完整矩阵 |
| 12/1 | 接口冻结 |
| 12/10–12/28 | 写论文、合作者内审 |
| 12/29–1/4 | 匿名代码包、提交 |

## 9. 需要确认的决定

1. 方法清单是否就是这 6 种，`Evolve` 是否纳入。
2. 5 个模型的选择（资格检查后给出候选）。
3. Beacon 算力预算上限（GPU 小时），超出时按第 6 节降级。
