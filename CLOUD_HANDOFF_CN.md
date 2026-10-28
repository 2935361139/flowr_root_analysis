# 云端代码交付与本地运行

本次把云端新增工具同步到GitHub，原有FLOWR模型源码、权重和实验输入不替换。

## 现在只运行这个检查

下载 [新增代码总包](https://github.com/2935361139/flowr_root_analysis/raw/refs/heads/main/cloud_handoff/flowr_cloud_code_all_v1.zip)，解压到一个新目录。

将其中的 `inspect_flowr_checkpoint.py` 复制到原项目根目录（与 `pyproject.toml` 同一层）。如果本地已有同名文件，先比较或备份自己的修改，不直接覆盖。

在原来的WSL终端执行：

```bash
cd /home/ljl/projects/flowr_root
.venv/bin/python inspect_flowr_checkpoint.py
```

使用现有 `.venv`，无需重装PyTorch或登录VS Code的Codex。只读检查权重哈希、检查点中实际保存的配置、源码版本和指定案例记录；不运行生成模型，不启动机制实验。

终端会打印 `outputs/checkpoint_audit_日期_时间_随机后缀/checkpoint_audit_result.zip` 的完整路径。把这个结果ZIP上传回聊天即可。即使状态是 `completed_with_issues` 也上传；它记录缺项或读取问题，不表示生成实验失败。若没有ZIP，保留结果目录并复制全部终端输出。

## 文件说明

| 文件或目录 | 用途 |
|---|---|
| `inspect_flowr_checkpoint.py` | 当前本地检查入口；单文件即可运行 |
| `checkpoint_audit_info/` | 中文说明、检查工具测试和验证记录 |
| `scaffold_study_v1/` | 此前的报告整理、末次处理对照工具及测试 |
| `scaffold_study_v1/mechanism_case_review/` | PTP1B单案例审阅证据；并非已完成内部机制验证 |
| `cloud_handoff/flowr_cloud_code_all_v1.zip` | 当前新增代码的便携总包，不包含原仓库依赖、权重或大型轨迹 |
| `cloud_handoff/scaffold_terminal_control_100_results.zip` | 已完成的100条末次处理对照结果，直接阅读即可 |
| `cloud_handoff/scaffold_study_100_results.zip` | 此前的100条报告整理结果，属于历史交付 |

旧文档中的实验计划保留作历史记录，不是当前执行清单。当前研究范围仍是 `ptp1b_lig3_seed_47` 一个案例；先核对原训练设置，之后再决定是否准备重放和干预入口。完整重放和内部机制干预仍为0/24；此前100条末次处理对照属于另一个已完成实验。

## 下载说明

上述 `cloud_handoff` 中的小型ZIP采用普通Git存储；直接链接下载实际压缩包，不依赖Git LFS。
原仓库其他路径仍有Git LFS数据。因此，目前不需要重新下载整个仓库，也不要用整个仓库ZIP覆盖你本地项目、`.venv`、权重或已有输出。

## 开发验证

```bash
.venv/bin/python -B -m unittest discover -s checkpoint_audit_info -v
.venv/bin/python -B -m unittest discover -s scaffold_study_v1/tests -v
```

这些软件测试不消耗生成实验预算。检查工具的正向检查点测试使用合成小检查点；你的原权重必须在本地检查，不能把测试通过当作已读到原训练配置。
