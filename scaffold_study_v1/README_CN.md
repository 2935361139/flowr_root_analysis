# scaffold_study_v1：统一诊断入口

当前本地操作请先看项目根目录的 `CLOUD_HANDOFF_CN.md`，只运行检查点检查。下述报告整理与末次对照属于已有工具；旧实验计划不自动继续执行。

新增：本轮末次处理的100条配对对照已由 `terminal_control.py` 实现并实际运行。100条基线复现全部通过，原生末次约束使PoseBusters通过数从58变为40，另有3条变为化学无效。使用方法及证据边界见 `TERMINAL_CONTROL_CN.md`；完整数据见另附结果包中的 `REPORT_CN.md`。
下文的报告整理入口仍可使用；旧 `final_constraint_plan.json` 只是历史计划，不是新入口的执行清单。

这次只新增本目录。旧源码、旧记录、旧评价器和旧实验包不替换、不删除。
原因：旧实验通过 SHA256 校验回放环境；覆盖旧脚本会破坏历史复现。

## 原有报告整理入口做什么

把已有“质量评价 → 轨迹观察 → 几何核查”接入一个入口，输出统一的逐样本证据。
核心逻辑与 FLOWR 读取逻辑分开；核查未知值、缺失帧、原子对应、批次身份和输出覆盖。
这个报告整理入口不重新推理、训练、修改模型参数或运行因果干预。新增的 `terminal_control.py` 则执行有限的末次处理干预，两者作用不同。

你要的完整目标仍包含神经网络机制研究。本版是其共用证据层，不是已经完成的自动根因解释器。
它可报告已支持的观察窗口；碰撞时间、其他未知错误和隐藏模块没有证据时明确保留“未定位”。
不新增一个训练解释器，不要求每个样本都被解释。

## 文件怎么放

解压后目录应为：

```text
你的FLOWR项目/
  flowr/                        原有，不动
  outputs/                      原有，不动
  scaffold_geometry_audit_v1/    原来的几何核查包（若已解压）
  scaffold_study_v1/             本次新增的整个目录
```

不要把目录里的 `core.py` 单独放到项目根目录。
如果项目已经有本目录，先比较版本和本地改动，不覆盖已有修改。
报告整理入口仅用 Python 标准库（Python 3.10+）；新增末次对照和原生操作测试使用原来的 FLOWR Python 3.12 环境及已有依赖。

## 运行：现有 100 条记录

在原来的 FLOWR 项目根目录执行：

```bash
python scaffold_study_v1/run.py \
  --quality-report outputs/sh_discovery_cohort_v2/quality/quality_report.json \
  --diagnostic-dir outputs/sh_discovery_cohort_v2_diagnostics \
  --geometry-dir scaffold_geometry_audit_v1/results \
  --output-dir outputs/sh_study_v1
```

`--geometry-dir` 指向含 `summary.json`、`protocol.json`、`sample_summary.csv`、`clash_pairs.csv` 的旧核查结果目录。
如果你原来的结果在 `outputs/sh_geometry_audit_v1`，只把这一项改为该路径。
如果只有 `scaffold_geometry_audit_v1.zip`，将原包解压到项目根目录后使用上面的命令；不要重复跑 GPU 生成或几何核查。
若暂时没有几何结果，可以省略该参数；程序仍统计 PB 未通过，但会明确说没有原子对定位。

预期：

```text
Completed: True; 100 runs; 10 input groups
Causal interventions: 0; root causes: not established
Review: .../outputs/sh_study_v1/study_review.zip
```

`Completed` 仅表示本次报告处理完成。若出现 STOP，保留报错，不改哈希、不删异常样本。
输出目录已经存在时，换新名称（例如 `outputs/sh_study_v1_check`），程序不会覆盖。
若文件内容是 `version https://git-lfs...`，那只是 Git LFS 指针，必须取得对应真实文件。

本次已在云端读取你的仓库保存报告，完成 100 条与早期 30 条的实际导入验证。
随交付提供的结果是这些报告的重新整理，不是重新生成的分子；你不必为了“取得结果”再生成一次。

## 看哪些输出

1. `REPORT_CN.md`：总览，先看它。
2. `findings.csv`：逐项问题、原子位置、观察窗口、证据等级。原始和修复后两个阶段分开。
3. `diagnosis.json`：完整结果，含原生时间、采样阶段、末两帧固定原子位移、参考碰撞和未定位项。
4. `sample_inventory.csv`：包括正常样本的全部输入清单，不只保留失败。
5. `normalized_evidence.json`：模型适配器与共用分析逻辑之间的接口。
6. `provenance.json`：输入、代码和输出 SHA256，记录本次证据来源。
7. `study_review.zip`：上述结果的压缩包，用于后续审阅。

未知不等于通过，也不等于失败。骨架未改变是任务指标，不是成键错误。
固定片段的坐标容差尚未预先规定，因此不输出一个冒充完整成功率的合成指标。
参考也不通过同名 PB 检查，不代表生成分子没有新增异常。

## 如需保存已讨论的末次约束对照计划

这条旧命令仍只分析报告并保存历史计划。当前已实现的100条离线末次对照请使用 `TERMINAL_CONTROL_CN.md` 的入口：

```bash
python scaffold_study_v1/run.py \
  --quality-report outputs/sh_discovery_cohort_v2/quality/quality_report.json \
  --diagnostic-dir outputs/sh_discovery_cohort_v2_diagnostics \
  --geometry-dir scaffold_geometry_audit_v1/results \
  --plan-final-constraint bace_lig2 ptp1b_lig2 \
  --max-continuations 40 \
  --output-dir outputs/sh_study_v1_plan
```

生成 `final_constraint_plan.json`：20 个原有种子全部保留，每个基线重放一次、约束开启一次，共40条完整生成任务。
预算不足时停止，绝不通过删除正常对照来凑预算。计划中每条状态都是 `not_run`。
若执行旧计划所描述的完整生成回放，仍需要原始完整张量、匹配权重和GPU回放执行器。当前末次处理实验复用保存的预测，不需要权重或GPU。
现有 recorder 没有通用的 `--final_inpaint` 命令行参数；不要自行给它添加一个不存在的参数，也不要把计划当成已运行结果。

## 开发、测试与后续适配

```bash
python -m unittest discover -s scaffold_study_v1/tests -v
```

- `flowr_adapter.py`：报告整理入口中读取 FLOWR 原生字段的适配器。
- `core.py`：共用异常分流和时间观察，不含 FLOWR 模块、层号或特定样本。
- `protocol.py`：有限实验清单与停止条件，不实施干预。
- `run.py`：输入检查、报告和可选计划入口。
- `terminal_control.py`：本轮 FLOWR 原生末次处理的离线配对对照，不是跨模型的通用执行器。

新模型需编写自己的适配器，输出 `SCHEMA_CN.md` 定义的格式，再用 `--normalized-input` 分析。
没有第二个真实模型的验证，不声称方法或机制已跨模型泛化。
换模型不要求存在 bond head；原生不提供某类证据时，留空并标明未覆盖，而不是造出概率或用距离冒充预测键。

下一阶段的研究顺序和停止规则见 `EXPERIMENT_RULES_CN.md`；proposal 可直接参考 `PROPOSAL_REVISION_CN.md`。
