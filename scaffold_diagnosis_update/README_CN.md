# 本轮：补齐现有失败的空间和时间定位

本包修改原有分析程序。没有重新生成、改模型、改评价阈值、网络扫描或干预。

## 运行

把压缩包解压到项目目录，得到 `scaffold_diagnosis_update` 文件夹。不要直接把子目录手工覆盖到项目中；使用安装器备份并替换。

```bash
cd /home/ljl/projects/flowr_root
.venv/bin/python scaffold_diagnosis_update/install_update.py --repo .
.venv/bin/python scaffold_diagnosis_update/run_update.py --repo .
```

默认处理 `outputs/sh_restart_v1` 的20条新记录，需要原来的完整记录（包括 inputs.pt）、quality、diagnostics，以及原始蛋白文件。使用现有 .venv，不需要安装新软件。

安装器只更新包内两个分析目录的文件。原文件备份到 `archive/diagnosis_backup_时间/`；如需撤销，把备份中同名文件放回原位置。新增加的 `scaffold_geometry_audit_v1/temporal.py` 可在撤销后删除。生成器、权重、评价器和旧结果不修改。

最后显示 `Completed: True` 表示这次分析完成，不表示分子通过或者原因得到证明。上传：

`outputs/sh_restart_v1/localization_v2/localization_review.zip`

若显示 False，也上传同一个包。若在创建输出前报错，提供终端文字。已有 localization_v2 不会覆盖或自动重跑。

## 修改内容

- 几何适配器支持新批次的扁平目录，按 case/ligand_idx/seed 生成稳定身份，拒绝歧义目录。
- 保留原生 PoseBusters 判据，复核最终结构与参考结构的检查结果；蛋白、水、有机和无机辅因子使用同一处理流程。
- 对最终实际发生碰撞的原子对，回查两条已保存序列：网络预测和采样状态。读 inputs.pt 的真实 pocket_com 恢复原坐标，核对末帧与原始终态吻合；读取 CPU 张量，不加载权重、不执行网络。
- 记录首次观察、最后连续异常区间、前一帧及其可判断性。原子类型不同、缺失或者距离恰好处于数值边界时记未知，不把未知当通过。
- 汇总程序继续报告所有已有失败类别和骨架任务结果；接入几何时间证据，不把骨架未变或中间带噪状态擅自改成化学失败。
- 原有化学轨迹定位继续保留。体积、能量、键角等没有对应局部时间诊断的项目明确未定位，不为凑结论生成假原因。

## 必须保留的解释边界

时间追踪对象是“终态这一原子对”，不是整分子的所有历史碰撞。当前原子类型不同于终态时不比较同一化学身份。最终筛选的原子对用于探索，不是独立验证。

网络预测与采样状态的 index 分别保留；不能直接当成同一种步骤。采样快照把积分与固定约束合并记录，因此不能进一步声称两者中究竟哪一个负责。此处不输出未经检查的网络层归因。

原始终态与解码后结果分开。只有原始终态碰撞对回查生成轨迹，解码后新增的异常保留为解码阶段现象。

这次完成后，依据实际定位结果固定最多3个候选/失败类别，再接入已有完整续跑工具做干预与对照。本包没有接入干预执行器，也不会自动追加分析轮次。

## 验证

11项新增软件测试、7项原有区域/映射测试通过。用上传的20条真实质量/诊断记录运行更新后的 study，未提供新几何信息时完整 diagnosis.json 与原结果一致。

当前开发环境没有 RDKit、PyTorch、原蛋白和完整 inputs.pt，未执行完整 PoseBusters+轨迹坐标定位。真实原生检查和坐标一致性由上述本地运行进行；不一致则报告错误并阻止后续汇总，不放宽阈值。

查看 geometry/clash_pairs.csv、geometry/temporal_evidence.json、study/findings.csv、study/diagnosis.json。只需上传一个 localization_review.zip。
