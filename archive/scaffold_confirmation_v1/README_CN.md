# 下一步：固定候选位置，换参考轨迹确认

本次不是重新做 198 次扫描。固定第一次网络调用（call=0）、第 9 层，用尚未用于这轮干预扫描的 6 条参考轨迹，检查之前的修复是否依赖参考轨迹。继续使用原来的生成、干预和评价代码。

## 如何运行

把整个 `scaffold_confirmation_v1` 文件夹放到 `/home/ljl/projects/flowr_root/`。原项目根目录中的 `resume_scaffold_scan.py` 和 `scaffold_causal_scan_v1` 保留。

```bash
cd /home/ljl/projects/flowr_root
source .venv/bin/activate

python scaffold_confirmation_v1/run_causal_plan.py \
  --plan scaffold_confirmation_v1/bace_confirm_plan.json \
  --output-dir outputs/sh_causal_confirm_bace_lig16_v1
```

使用前一轮相同的环境、模型、记录路径。默认读取 `outputs/sh_head_bace_lig16_v3` 和 `outputs/sh_boundary_screen_v1/bace_lig16`。若这些目录确实已移动，分别用 `--head-dir` 和 `--output-root` 指向原始完整记录。

正常结束应显示 `Completed: True; 33/33`。上传：

```
outputs/sh_causal_confirm_bace_lig16_v1/causal_scan_review.zip
```

如果中断，在同一条运行命令最后加 `--resume`。已完成结果须通过校验才会复用。旧的 198 次结果仍在旧输出目录中。

## 这 33 次具体是什么

- 7 次原始生成复现：失败轨迹 50，以及参考轨迹 45、46、47、48、49、51。
- 2 次不改变表征的对照。
- 12 次定向替换：6 个参考 × 原子/原子对两种表征。
- 12 次同等幅度随机方向对照。

主要比较同一候选位置的 6 次原子表征替换与其 6 次随机对照。原子对表征作为另一种表征的对照，分别报告，不混成总成功率。

六个参考种子来自此前已有记录，并非根据这次干预结果挑选。七条基线仍须通过原有的复现和固定片段条件检查；若校验不通过，应检查原因，不得跳过。

## 如何决定后续

1. 报告全部参考轨迹的结果，不仅看成功案例。同时检查化学有效、固定片段、PoseBusters 和新骨架是否变化。
2. 如果在另外多条参考轨迹下仍出现定向修复，而且与随机对照有区别，再预先限定第 9 层附近及其内部模块的定位实验。
3. 如果效果仅限原来的两个参考，应把结论限制为“依赖参考轨迹的局部干预效应”，先分析差异，不能继续修改位置直到碰到成功。
4. 对独立失败样本、其他分子使用固定分析规则，才是在检验跨样本泛化。这一轮仍只有一个失败样本，不承担该结论。

`False` 表示没有达到该列对应的指标；它不等于程序失败。空值表示该指标未能评价，不是合格。看到 `Completed: True` 表示程序完成，不表示全部分子合格。

## 验证范围

本包复用已实际运行完成的 v1 科学计算程序和 JSON 修复。新入口只支持新输出目录、明确配置和原有断点续跑，保留模型/输入/源码/随机流/结果文件校验。交付前检查计划计数、源码匹配、真实缓存的序列化、启动和续跑路径；本地没有模型 GPU 环境，尚未实际执行本轮 33 次生成。
