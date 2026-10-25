# 下一步：追踪早期内部变化

目的：对已有 BACE ligand 16 的 10 条轨迹，用同一套记录方法检查最初 5 次网络调用。
看清固定连接区域的几何、内部表征、内部键与外接键如何一起变化，再决定值得检验的机制。

这是 proposal 中“沿真实生成轨迹定位候选关键计算”的阶段。此前尚未确立稳定根因。
本步只回放已经保存的状态，不重新生成分子，不训练模型，不改参数或隐藏表征。
已有终态评价标准不变，也不要求新骨架恢复成参考骨架。

## 你只需要运行一次

把 `scaffold_early_observation_v1.zip` 放到 `/home/ljl/projects/flowr_root/`。
在 VS Code 的终端或服务器终端执行：

```bash
cd /home/ljl/projects/flowr_root
source .venv/bin/activate
python -m zipfile -e scaffold_early_observation_v1.zip .
python scaffold_early_observation_v1/run_early_observation.py \
  --output-dir outputs/sh_early_bace_lig16_v1
```

不用手动修改旧脚本，也不用删除旧输出。包内四份适配代码复用之前已核对的版本。
新入口把它们合并在一次模型加载中。运行时间取决于 GPU；这里没有测定你的 GPU 耗时。

默认读取已有目录：

```text
outputs/sh_boundary_screen_v1/bace_lig16/
  sh_record_bace_seed42/research_record/...
  ...
  sh_record_bace_seed51/research_record/...
outputs/sh_head_bace_lig16_v3/summary.json
```

每个原始记录目录必须保留完整的 `forward_calls`、`sampler_steps`、`inputs.pt`、
`pocket_encoding.pt`、`raw_final_prediction.pt` 和任务/运行元数据。此前上传的精简 review ZIP 不能替代它们。

## 运行结束看哪里

完整成功时，终端最后显示：

```text
Capture completed: True
Analysis completed: True
Completed: True
Accepted calls: 50/50
Upload: .../early_observation_review.zip
```

50 是 **10 条已有轨迹 × 5 个保存状态**，不是生成 50 个新分子。
每个状态先无钩子回放，再带只读钩子回放，因此约 100 次网络前向计算。
所有调用与层的原始索引从 0 开始；表格中的 `layer` 从 1 开始，另有 `layer=0` 表示第一层入口。

请上传：

```text
outputs/sh_early_bace_lig16_v1/early_observation_review.zip
```

也可打开同目录的 `REPORT.html` 查看图和表说明。
若出现 `False` 或 `STOP`，先上传同一个 ZIP；它会保留已收集的记录和错误文本。
`False` 表示某项程序检查未通过，不等于又发现一种分子错误。
不要绕过一致性检查继续分析。

程序不覆盖已有输出目录。如果这个目录以前已经运行过，请保留它，改用一个新的输出目录名。
如果只有绘图/分析失败而采集已完成，可在核对错误后用以下命令重新分析，不加载模型：

```bash
python scaffold_early_observation_v1/run_early_observation.py \
  --analyze-only --output-dir outputs/sh_early_bace_lig16_v1
```

## 本批数据回答三个具体问题

1. **最先变的是模型预测，还是实际采样状态？**
   同时记录网络输入、自条件、未经后处理的网络输出、积分及固定片段约束处理后的状态。
   区分“模型预测的固定原子距离变长”和“实际固定原子被移动”。初始全零自条件标为未知。
2. **变化从哪些层、哪些子模块出现？**
   记录所有解码层的几何读出、自注意力、口袋条件注意力、前馈更新及中间坐标重置。
   另记录两个固定原子的标量表征和连接位点的原子对表征。没有预先锁定第 9 层。
3. **内部键和向新骨架接出的键是否发生相互冲突的变化？**
   记录连接位点全部有效配对及所有键型概率。既看固定片段内部键，也看新骨架外接键；
   同时保存原子类型与电荷类别输入/输出值，便于检查类型变化。

我们会先看这批统一结果，再提出明确的候选机制及后续干预预测。
若没有一致证据，就明确保留“尚未定位”，不会把最大的范数、最大的得分项直接命名为根因。

## 输出文件如何读

| 文件 | 用途 |
| --- | --- |
| `pair_states.csv` | 各阶段的距离、原子类别、键类别和全部键型概率 |
| `site_states.csv` | 固定连接位点外接了哪些生成原子，及描述性的键级总和 |
| `final_focus_status.csv` | 最终焦点键是否保持；不是整体分子是否合格 |
| `layer_pair_observations.csv` | 层内坐标和原子对表征变化 |
| `scalar_branch_writes.csv` | 两个固定原子的各子模块标量更新范数 |
| `bond_head_changes.csv` | 相邻调用中，成键头得分变化的节点、几何、原子对及数值项分解 |
| `coordinate_changes.csv` | 最终坐标读出的完整变化核算 |
| `coordinate_submodule_changes.csv` | 上表“隐藏向量变化项”的层/子模块分解 |
| `seed_*/call_*.npz` | 可复核的原始内部记录 |
| `summary.json` | 原始文件身份、配置、逐次一致性检查和完成状态 |

键类别：0 无键、1 单键、2 双键、3 三键、4 芳香键。
固定原子编号来自已核实的任务映射；不同随机种子的生成原子编号没有身份对应关系。
图的红/蓝分组只描述最终焦点键，不能称为“整体失败/成功”。
采样早期的噪声结构及键级总和只作过程记录，不能套用终态合法性标准判错。

## 结论边界与复用范围

- 分解能加回实际变化，是数值正确性的检查；不是因果证明。
- 范数和几何读出分解不能替代输入原子 SHAP，也不能直接解释为某原子的贡献比例。
- 这批仍是同一复合物的探索数据，不是独立验证集。另一个复合物是否有同样机制，需要后续验证。
- 计划文件可配置种子、调用窗口和固定连接位点；统计与记录代码不依赖 seed 50 或第 9 层。
- 当前适配器明确面向已检查的 ROOT 源码、单分子回放、Euler 求解器、每积分步一次前向调用。
  换模型、模型版本或采样控制流，需要核对适配器；程序会拒绝不支持的情况，而不是静默套用。
- 本批只查看指定固定连接区域及其配对，不能声称自动诊断了分子中的所有可能错误。

## 已做的程序检查

已检查 Python 语法和命令入口；用真实 ROOT 层控制流的 NumPy 模拟测试，
覆盖不同原子数、通道数、焦点编号和条件/坐标更新组合，核对五组观察钩子同时运行不改变输出、
残差分解成立、错误记录会被拒绝，并完成模拟数据的 CSV/图片/HTML/ZIP 生成。
另外复算了此前真实上传的成键头记录。结果见 `VALIDATION.json`。

本环境没有 PyTorch、RDKit 和你的预训练 GPU 模型，**没有在这里执行实际 GPU 回放**。
运行时必须逐次通过原模型预测一致性和原生成环境检查；未通过的记录不会用于研究结论。
测试用模拟数据没有打包为实验结果。
