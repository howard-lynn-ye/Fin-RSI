# Claude 接续任务：fin-skills 效用实验与论文交付

用户于本轮要求 Codex 先收尾，改由 Claude 接续。以下是实际状态，不是全部任务完成声明。
工作目录 `D:/fin_skill`；分支 `codex/library-reliability-20260928`；现有草稿 PR：
https://github.com/howard-lynn-ye/fin-skills/pull/11 。继续更新这个 PR，不要重复开一个。

## 首先执行

1. 读根目录 `AGENTS.md`、本文件、`paper/CODEGEN_UTILITY_PROTOCOL_20260928.md`。
2. 检查本地完整测试日志 `runs/codegen-full-pytest-20260928.log` 是否已经结束。
   交接准备时仍在运行；不要与生成包命令并行，也不要先重复启动整套测试。
3. 查询 Beacon 作业 **1733007**，不要重新提交或取消它。
   最新成功观察为 RUNNING，已完成 **24/48 份模型回答**，Linux 沙箱资格检查已通过。
   SSH 偶发 banner EOF；这不是模型失败或作业失败。连接失败时先做其他任务，稍后有限重试。

```powershell
python scripts/beacon_workspace.py run --credential-file D:/ct_agent_vqa/.secrets/beacon_password.txt --remote-root /beacon-projects/radfm/wy891/fin-skills-campaign-codegen-utility-20260928-v2 --command 'squeue -j 1733007 -o "%i %T %R"; tail -n 5 logs/slurm-1733007.out; tail -n 8 logs/slurm-1733007.err'
```

凭据只通过已有文件供传输工具读取，禁止打印、提交或上传。已验证固定主机指纹。
`scripts.beacon_campaign.connect()` 和 `scripts.deploy_beacon_followup.remote_command()` 可复用；
用 SFTP 一次取多个文件比反复建立连接稳妥。本轮快照脚本是
`runs/snapshot-codegen-handoff.py`，原始数据下载脚本是 `runs/fetch-finqa-reanalysis.py`。
这两个脚本仅在本地 `runs/`，没有提交，现成凭据也不在仓库中。

## 已完成的前一阶段

- Pandas 月末/季末/营业季末别名改为跨版本 offset 对象，保留数量和财政季度锚点；
  包依赖下限及最低版本 CI 为 pandas 2.1.4。
- 前一轮完整测试：2.1.4 为 3493 passed；2.2.2 为 3426 passed；3.0.6 为 3488 passed，
  均无失败。可选依赖 skips 不同，默认 deselect 53 个慢测试。准确范围见
  `paper/LIBRARY_RELIABILITY_CORRECTION_20260928.md`。
- 论文已转回 fin-skills、Guards 和数值可靠性，HiSTrim/Memory 为辅助。
- 原六个修订 TeX 已同步 Overleaf，源码回读 SHA256 一致，云端 0 errors/0 warnings；
  本地 PDF 13 页。这是**上一轮**结果，尚不含本轮的新代码生成实验或格式重分析。
- 前一轮提交：`861033a305d87a3968400e6e58bc5e4e856eb45e` 和 `c608201`。

## 本轮新实验 A：正在实际运行

代码：`benchmarks/agent_study/codegen_utility.py`、`codegen_worker.py`。
已测试独立 NumPy 评分公式与库组合一致，及配对输入/解析规则，共 **14 passed**。

设计已在模型调用前冻结：Qwen2.5-Coder 7B/14B × 四任务 × 三 seed × 两条件 = 48。
任务为对角风险平价（逆波动率，不是完整 ERC）、横截面动量、均线交叉和波动率目标动量。
同一对使用相同合成拆股数据、数学定义、时间约束、单轮 2048 token 上限和生成 seed。
库条件额外得到真实 catalog 返回及 API/Guard 文档，并生成实际调用库的 Python。
原始条件从头写 NumPy/pandas/SciPy。没有错误反馈和挑选重跑。

全部回答冻结后，作业会自动评分：首份程序可执行性、独立公式误差、含同日依赖的未来
扰动、拆股表示不变性、真实库/guard 调用轨迹、token 与时间。程序使用已有 Landlock /
seccomp 沙箱。两组均接受同一事后评分和实际 causality guard，代理主动调用 guard 单列。
这是“目录/文档/库支持”的配对比较，不是 guard-only 因果实验，不承诺结果一定有利于库。

远端目录：`/beacon-projects/radfm/wy891/fin-skills-campaign-codegen-utility-20260928-v2`。
重要产物：`protocol.json`、`inputs.json`、`qualification.json`、`responses/`、
`inference-receipt.json`、`audits/`、`scores.json`、`completion.txt`、`logs/`。
原代码及库快照在 `source/library-and-runner.tar.gz`；哈希在 `manifest.json`。

本地已提交的冻结记录在 `benchmarks/agent_study/evidence/20260928-codegen-utility/`。
**还没有最终模型结果，不要把提交成功、24 份回答或沙箱通过写成实验完成。**
v1 的 shell 首行多了一个空行，被 sbatch 拒绝，未运行推理。v2 仅修复 shebang；拒绝记录已保留。

完成后下载全部 responses/audits/receipts/scores/logs 到该证据目录，并逐项校验 SHA256 和 48 个
计划单元齐全。分别报告每个模型每组 12 个程序的结果；金融有效率不能用“代码执行成功”代替。
若发生基础设施错误，保留原始尝试；修复评分器不允许重新生成或覆盖已有模型回答。
审查 frozen source 与当前源码一致性，不要修改正在执行的目录。

## 本轮新实验 B：冻结 FinQA 格式重分析已完成

代码：`benchmarks/agent_study/finqa_format_reanalysis.py`，四项边界测试通过。
完整原始下载在 `runs/finqa-format-reanalysis-20260928/` 和同名 `.tar.gz`（本地保存）。
跟踪证据：`benchmarks/agent_study/evidence/20260928-finqa-format-reanalysis/`。
`extraction-receipt.json` 先绑定 192 份目标无关提取结果，随后才由官方 executor 和 gold 评分。
所有选定 episodes 哈希匹配原 inference receipts，原始六组官方正确数复算一致。

| 模型/条件 | 原严格程序正确 | 单层 fence v2 正确 | 可恢复最终答案正确 | 缺失最终答案 |
| --- | --- | --- | --- | --- |
| Qwen RAG | 2/32 | 2/32 | 3/32 | 3/32 |
| Qwen BGE | 2/32 | 2/32 | 2/32 | 3/32 |
| Qwen Kev | 0/32 | 0/32 | 2/32 | 3/32 |
| Mistral RAG | 0/32 | 0/32 | 0/32 | 26/32 |
| Mistral BGE | 0/32 | 0/32 | 0/32 | 24/32 |
| Mistral Kev | 0/32 | 0/32 | 0/32 | 25/32 |

“可恢复最终答案”是新列：原可解析程序正确，或最终自然语言中按冻结规则提取到明确正确数字。
它不是程序正确率，也不证明引用或检索正确。百分号归一化；歧义、拒答和缺失仍失败。
没有用最后一次计算器输出替代缺失答案。旧结果未改。这是事后敏感性分析，不是新的推理实验。
新 ReAct/JSON 提示及版本化解析已在前一轮修好，但新的真实模型 FinQA 对比尚未运行。

重要结论：格式不是唯一原因；fence 放宽没有新增正确程序。不能声称“分数全因 JSON”或
“修好格式已经证实重排收益”。如后续需要新推理，应先在开发题验证正常结束，再采用同提示/
同 parser 的匹配比较；不得把新一组与旧控制组包装为匹配实验。

## Claude 应完成的收尾

1. 等待并审计上述现有 GPU 作业。必要时修复真正的基础设施故障，保留失败记录。
2. 将真实配对结果及 FinQA 重分析加入论文主表/相关附录；改摘要、引言、贡献、讨论、结论。
   主角是 fin-skills 的专业计算与防缺陷效用；不要为了积极叙述杜撰“显著提升”。
3. 数据边界：36/36 是 12 类植入缺陷 × 三世界，非 36 类泄漏；8/8 通过各自容差，最大
   绝对误差约 1.18e-8，不是全部 <1e-8；未验证的 71.5 假 Sharpe 不能直接作为实测数字。
   经验运行时拟合也不是 O(N^0.26) 的理论保证。原 FX 无强制 guard 实验保留为组件消融。
4. 运行根目录要求的三个生成/验证步骤和与最终变更相称的测试。当前完整 pytest 正运行，
   先读其日志。之后新增的 FinQA 四项测试单独已通过；不要因为接力重复耗时测试而拖延主线。
5. pdflatex + bibtex + 两遍 pdflatex，检查 undefined citations、溢出、主文页数与关键页面。
   已查 ARR 官方作者规则：长文正文上限 8 页，不含 limitations/ethics/references/appendices：
   https://aclrollingreview.org/authors 。这是篇幅核验基准，不等于已经确认具体投稿资格。
6. 同步现有 Overleaf：https://www.overleaf.com/project/6aad9a03f27c3c07a182965d 。
   先读取云端，保护合作者独立编辑。云端 OUTLINE.md 与本地不同，不要全量覆盖。
   更新实际需要的 TeX，云端编译，逐文件回读校验 SHA256。写 `paper/SYNC.md` 和 JSON 同步回执。
7. 提交并推送当前分支，更新 PR #11 标题/正文，交付完整完成回执。不要自动合并。

Overleaf 浏览器经验：源文件下载偶发被浏览器阻止，可在 Source Editor 中全选、复制全文后
计算 LF 规范化 SHA256。切换文件后必须先确认编辑器已显示新文件，再复制，不能立即连续批量
操作，否则会读到上一文件。前次标签为 `Fin-skills library reliability - 2026-09-28`。

用户已授权以上实验、代码修复、论文编辑及既有 Overleaf 同步，无需再次问能否继续。
