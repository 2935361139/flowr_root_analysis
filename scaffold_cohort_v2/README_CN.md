# 新输入发现批次：scaffold_cohort_v2

## 这一步解决什么

用同一套规则收集更多不同配体输入的成功与失败轨迹，判断连接失败是否在不同输入中反复出现。不是训练模型，也不是根因检验。

默认 10 个未用过的、结构去重的合格配体输入，每个 10 个随机种子，共 100 次生成。输入来自当前 examples 的 CDK2、TYK2、PTP1B、BACE 四个蛋白示例；不等于 100 个独立复合物，也不代表新蛋白、未见训练数据或跨模型泛化。没有匹配失败也是合法结果，不无限追加种子寻找想要的结果。

保留成功和失败数据。选择只查看输入结构和旧输入记录，不根据生成结果筛选。排除本地 outputs 中已有记录的输入和可识别的相同配体结构；明确排除已知 BACE16、PTP1B0、CDK2 1、TYK2 3、PTP1B1。历史记录已删除时，不能保证识别所有曾经用过的输入。

## 放在哪里、需要恢复什么

把 ZIP 解压到 FLOWR.ROOT 项目根目录，只增加 scaffold_cohort_v2 一个文件夹。

复用已有 flowr/、checkpoints/flowr_root_v2.2.ckpt、examples/ 和 Python 环境。原有 outputs/ 保留。

本包包含运行所需的 record_scaffold_run.py、inspect_scaffold_trace.py、quality.py 三个旧辅助程序的原样副本，不依赖根目录的同名文件，不需要恢复过去所有分析脚本。源码和辅助程序哈希与已审核版本核对。截图看不到子文件，故 prepare 仍需检查本地真实依赖。

*.py:Zone.Identifier 是下载附带信息，不是 Python 程序。不要把它重命名成 .py。

## 在 VS Code 的 WSL 终端执行

先使用此前成功运行 FLOWR.ROOT 的 Python 环境，再进入项目根目录：

```bash
cd /home/ljl/projects/flowr_root
python -m zipfile -e scaffold_cohort_v2.zip .
python scaffold_cohort_v2/run_batch.py --prepare
```

应看到 `PREPARED: 10 inputs x 10 seeds = 100 baseline runs; 0 interventions.`
这一步不加载生成模型、不运行 GPU 推理；会读取并校验权重文件、输入、历史记录和依赖。

若合格新输入不足、历史记录损坏、源码或权重不匹配，程序停止，不改阈值、不覆盖历史输出、不把程序错误标成分子失败。查看 STOP 信息及新输出中的 input_inventory.json / batch_status.json。部分前置依赖失败发生在输出目录建立之前，此时以终端信息为准。

先完成 1 次，确认当前机器可以记录完整轨迹：

```bash
python scaffold_cohort_v2/run_batch.py --run --max-new-runs 1
```

应看到 `PAUSED at requested run limit.`，batch_status.json 中 complete 为 1。它是同一批 100 次中的第一次，并不是额外数据。此时只完成生成和轨迹完整性检查，尚未进行整批质量评价。

检查磁盘空间及第一次记录的大小，完整批次需保存 100 份轨迹；不足时先停止，不删除旧实验记录。然后继续同一计划：

```bash
python scaffold_cohort_v2/run_batch.py --run
```

已完成的记录会检查后复用。程序不会覆盖半途失败的记录；出现 STOP 应保留日志和记录以排查，不能直接删掉再换种子。

## 输出

全部写入 outputs/sh_discovery_cohort_v2/，不覆盖旧批次。

- frozen_plan.json：固定的输入、种子、源码、权重、环境信息。
- runs/：原始结果、未经价态修复的解码、轨迹张量、原子映射、过程日志。
- quality/quality_report.json：两种解码阶段的详细评价。
- baseline_summary.json / baseline_labels.csv：原始未修复阶段的摘要。
- batch_status.json：运行状态；baseline_evaluation_complete 表示本批运行及评价完成，不表示分子全都成功。
- validation_review.zip：便于回传的 JSON、CSV、SDF、日志和程序副本；不含较大的 PT 轨迹和权重。请继续保留本地原始 PT 文件。

完成后只需提供 validation_review.zip。若中途停止，提供完整 STOP / traceback，或已生成的这个审查包。

## 评价的边界

继续使用旧程序里调用的 FLOWR 化学有效性、连通性、固定子结构匹配和安装环境中的 PoseBusters dock 检查，另报告映射后的固定元素、电荷、内部键是否保持、原始图连通性、固定坐标位移和骨架结构是否改变。

- 新骨架不同于参考不是错误；参考外连键完全相同不是通用成功条件。
- 固定坐标位移以连续数值报告，目前没有预先确定的通过阈值，不能宣称已满足全部固定几何条件。
- 原始解码与默认修复后的解码分开，避免把后处理改动误认为网络内部变化。
- 化学有效但未改变骨架，不等于成功骨架跃迁。
- 无效分子的部分三维/子结构评价不可执行，保留未评价状态和分母，不将所有空值强行转换为 False。
- 摘要的 chemical_connection_endpoint 仍不是完整三维 scaffold-hopping 成功指标，完整明细在 quality/。
- 不声称覆盖所有可能的错误，或已证明生物活性保持。

## 与旧批处理的改动

只改批次组织：3 个输入扩至 10 个；改用独立输出目录；去掉“连接位点旁必须存在固定内部键”的特定机制筛选；去掉预设 call1 反馈修复方案；增加单次运行后暂停和继续；摘要增加现有评价字段。三个辅助程序保持原样。没有改变模型权重、生成参数或修复规则。

## 这一步之后

先按相同规则统计不同输入的连接失败，并选定一种在多个输入中重复的连接错误。按输入划分发现和独立验证样本，不能把同一输入的不同随机种子当成跨输入验证。

之后才在成功/失败对照中进行“时间 × 子模块”的统一观察与受控干预。现有记录保存网络输入输出与采样状态，不记录每一隐藏层激活；后续通过校验一致的重放采集选定模块，而不是现在对所有失败追加专用脚本。

增加样本仅解决案例不足，不能替代候选机制、合适对照和完整后续生成。若失败只集中在一个输入，本批不能支持跨输入机制结论。

## 已做的验证

8 项 CPU 软件测试通过，覆盖输入去重/历史排除、缺失评价不当失败、无固定内部键但有连接位点的输入可入选、完成后复用、部分失败保留、程序/计划改动阻止运行、单次暂停后继续。测试中的模型和分子后端是明确的合成模拟，不是科学实验结果。

13 个 FLOWR 源文件及 3 个原样辅助程序的 SHA256 与配置一致。本环境没有 RDKit 和用户 GPU 权重，未在此完成真实分子批次或新的 GPU 推理；实际兼容性由用户环境的 prepare 和第一条真实记录检查。
