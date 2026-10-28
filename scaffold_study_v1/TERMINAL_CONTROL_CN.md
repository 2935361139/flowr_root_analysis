# 本轮限定任务：末次处理的100条配对对照

入口是同一目录下的 `terminal_control.py`。只执行已讨论的末次处理实验，不增加生成种子，不扫描神经网络。

## 做什么

读取原有100条记录中的：

- `inputs.pt`：固定片段初始状态、坐标缩放与口袋质心。
- `sampler_steps/step_0100.pt`：最终网络预测转成原生分布之后、恢复全局坐标之前的保存状态。
- `raw_final_prediction.pt`：原始最终输出，用于验证等价性。
- `task_definition.json`、`reference_preprocessed.sdf`、原始蛋白文件：映射与评价。
- 完整质量报告和诊断报告：核对输入、源码及原有结果。

不加载权重，不调用神经网络，不重新随机采样。需要原 FLOWR Python 环境，CPU 即可。
原始PT文件必须存在；仅有上次的 `study_review.zip` 不够。
云端已完成本轮运行，直接阅读另附结果包即可取得结果，不必为了看结果再次运行。

## 两个分支

1. OFF：保存的末次预测 → 原生坐标恢复 → 原生解码与质量评价。
2. ON：同一预测的独立副本 → 原生 `MolBuilder.inpaint_molecule` → 相同坐标恢复、解码与评价。

两分支均保留未经价态修复的解码和运行设置下的解码；它们不是两次独立生成。
原生 inpainting 除恢复固定坐标、元素、电荷与内部键之外，还会将整个键张量按上三角 argmax 对称化、独热化。
因此本实验测试的是完整原生末次开关，不能把其全部效果解释为“只恢复固定坐标”。程序记录固定内部键及其他键的类别变化、概率变化、生成原子坐标与种类变化。

## 基线验收

先验证全部100条OFF分支，通过后才执行任何ON分支：

- 坐标与保存的最终输出最大绝对误差不超过预先固定的 `1e-5 Å`，不使用相对误差。
- 原子、键、电荷分布及有效掩码精确一致，类别决策精确一致。
- 原始与修复解码的化学、连通、固定子结构、固定化学结构、骨架变化、分子图及全部PoseBusters布尔结果复现。
- 验证范围是分子结构与评价；不声称复现亲和力等辅助输出或整条神经网络轨迹。

输入的Git LFS文件若尚是指针，程序只读取已经下载且校验和匹配的本地LFS对象，不下载或修改仓库。
记录缺失、基线不一致或源码变化会停止。保留错误目录，排查后用新的输出目录重跑，不绕过检查。

## 本地复核命令

在原FLOWR项目根目录，激活原环境后执行：

```bash
python scaffold_study_v1/terminal_control.py \
  --output-dir outputs/sh_terminal_control_v1
```

默认读取：

```text
outputs/sh_discovery_cohort_v2/quality/quality_report.json
outputs/sh_discovery_cohort_v2_diagnostics/
outputs/sh_discovery_cohort_v2/runs/
```

位置不同可指定 `--quality-report`、`--diagnostic-dir`、`--records-root`、`--repo`。
保留原来的100条清单，不删正常样本或只留下失败。

## 输出

- `status.json`：必须同时满足 completed=true、baseline_runs_passed=100、paired_runs_completed=100、errors=[]。
- `results.json`：每个样本两个分支的所有指标及总表。
- `paired_summary.csv`：便于查看的前后对照。
- 每个样本目录：基线验证、对照结果、四份SDF及ON终态张量。
- `protocol.json`、`input_hashes.json`：固定规则、来源与校验和。
- `terminal_control_review.zip`：JSON/CSV/SDF审阅包；原始PT和ON张量留在原目录，不塞进审阅包。

无效分子的几何检查保留为未评价，不算通过，也不强制改成失败。
PoseBusters原生布尔值还需要结合日志解释：本轮 `tyk2_lig2_seed_48` 在OFF和ON、两种解码下都无法生成能量参考构象，工具将 `internal_energy` 记为False；不能据此写成已测得能量过高。结果包另附 `evaluation_caveats.json` 和运行日志，不改写原生返回值。
固定坐标相对于参考SDF的微小残差可能来自SDF小数截断；精确恢复检查另外针对原始张量执行。
修复一种指标但损害其他指标必须同时报告，不合成一个未预先定义的“完全成功率”。

## 软件测试

使用原FLOWR环境运行：

```bash
python -m unittest discover -s scaffold_study_v1/tests -v
```

本轮结束后停止。结果支持末次流程的作用，不直接说明哪个隐藏层产生了错误预测。没有新增样本、上游模块干预或事后消融。

本轮实际结果：100条、10组输入；全部分子张量基线误差为0，两种解码的原有质量标签全部复现。OFF/ON化学有效数为100/97，PoseBusters通过数为58/40。18条从通过变为未通过，0条从未通过变为通过；另3条因化学无效而不评价几何。结果不支持把末次约束直接作为整体修复方案。
