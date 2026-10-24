# 现有文件用途审阅

依据：这次文件截图、可读的已有脚本及依赖、已完成的 `quality_report.json`。截图不能证明某文件没有其他调用者，因此暂不批量删除或改路径。

## 保留为基础设施

| 文件/目录 | 用途及决定 |
|---|---|
| `record_scaffold_run.py`、`run_scaffold_pilot.py`、`inspect_scaffold_trace.py` | 生成记录、批量执行、记录检查；保留 |
| `scaffold_evaluation_v2/evaluate_scaffold_outputs.py` | 本轮已完成统一终态评价；保留，本次不必重跑 |
| `outputs/sh_record_*`、`outputs/sh_quality_multicase_v2` | 原始证据、任务定义、评价结果；保留 |
| `flowr/`、`checkpoints/`、`examples/`、`.venv`、`posebusters/`、项目配置 | 模型、输入、环境与运行依赖；保留 |

## 保留为模型适配工具，后续统一调用

`replay_connection_hidden.py`、`diagnose_scaffold_replay.py`、`trace_connection_head_v2.py`、`trace_connection_submodules.py`：回放、内部计算记录及检查工具具有复用价值。是否可以迁移到另一模型，取决于其结构与接口；不能仅凭文件名判断已经通用。

## 冻结为旧探索，不再逐个手动作为主流程执行

- `trace_connection_sites.py`：旧版严格边界诊断。与参考的外连键完全一致不再是通用失败标准；本包保留该信息，但不据此挑选失败。
- `analyze_connection_window.py`、`causal_scaffold_localization.py`、`trace_connection_rebound.py`、`trace_connection_earlier_window.py`、`test_bond_state_paths.py`：包含已有观察、回放、干预与负结果，先保留路径。
- `bond_paths_plan.json`、`causal_plan.json`、`earlier_window_plan.json`、`rebound_plan.json`、`ptp1b_plan.json`：用于特定样本/阶段的探索计划。后续应由统一规则生成计划，而不是每个样本手写一份。
- `outputs/sh_connections_ptp1b`、`sh_hidden_ptp1b_v2`、`sh_hidden_valence_window`、`sh_head_decomposition_ptp1b_v2`、`sh_coordinate_source_ptp1b`、`sh_observation_ptp1b`、`sh_submodule_localization_ptp1b`、`sh_causal_localization_ptp1b_v2`、`sh_connection_rebound_ptp1b_v1`、`sh_earlier_connection_ptp1b_v1`、`sh_bond_state_paths_ptp1b_v1`：旧结果保留，避免把未修复的实验结果当作无用文件删除。

“冻结”表示不再作为当前的日常入口，文件仍留在原位。部分脚本导入其他旧脚本并校验 SHA256；现在移动或修改旧文件可能破坏回放。后续完成依赖整合后，才把整组旧实验归档。

## 可以另行清理但不影响研究结论的内容

截图里的 `:Zone.Identifier` 是 Windows 下载来源元数据，通常不是实验数据。清理它们不会补齐方法泛化或内部机制证据。本包不执行清理。

## 方法层面需要改什么

过去的主要问题是把“终态评价”“针对一个样本的探索”“可复用的机制诊断”混在一起使用和解释，而不是每份代码都无效。日常入口需要收敛，失败类型和证据等级需要统一。

泛化应分三层检验：

1. 同类错误换样本、换原子编号，代码与规则不变；这是最基本的复用。
2. 机制解释换复合物仍能预测后续更新与干预效果；这需要独立数据验证。
3. 新错误类型应明确标为未覆盖，再增加一类通用处理方法；不能声称现有方法自动解释所有未知问题。

当前保留了基础设施和有价值的探索记录，本包先建立第一层的观察入口；第二层因果机制与跨复合物验证尚未完成。
