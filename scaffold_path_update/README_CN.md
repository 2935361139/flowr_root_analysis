# 按路径输入、分类与干预

这次修改的是输入输出与任务选择。沿用原来的生成参数、判别标准、定位方法和 A/B/AB/C 干预及对照。没有增加新的修复算法，原来不支持的化学结构错误、UFF 参数缺失等仍会如实报告。

## 安装一次

先等当前实验结束。把 `scaffold_path_update.zip` 解压到项目根目录，然后在原来的环境执行：

```bash
cd /home/ljl/projects/flowr_root
source .venv/bin/activate
python scaffold_path_update/install.py --repo .
```

会自动备份替换的文件到 `archive/validation_paths_backup_*`。安装兼容原 validation100 和上一版可调参数入口；遇到未识别的核心代码修改会在写入前停止，避免覆盖。

只修改**项目根目录下** `scaffold_validation100/config.py`，不要修改安装包 `payload/` 里的副本。旧的配置会随旧代码保存在备份中；首次升级使用下面的 5 个样本配置，务必检查路径。

## 每次只改 config.py

```python
SDF_PATH = "examples/cdk2_ligands.sdf"
PDB_PATH = "examples/cdk2_protein.pdb"
LIGAND_INDICES = [5]
SAMPLES_PER_INPUT = 5
SEED_START = 3001

GENERATION_DIR = "outputs/experiment_001/generated"
INTERVENTION_INPUT_DIR = "outputs/experiment_001/generated/failed"
INTERVENTION_OUTPUT_DIR = "outputs/experiment_001/intervention"

INPUTS_PER_GROUP = 2
```

- `SDF_PATH`：输入配体文件，可换成你自己的 SDF 路径。
- `PDB_PATH`：对应蛋白。SDF 配体需要是该蛋白坐标系中的三维构象，不能拿任意二维分子或另一个口袋的坐标直接替换。此次一批使用一个 SDF/PDB 对；不同蛋白分批运行即可。
- `LIGAND_INDICES = [5]`：取 SDF 中编号 5、也就是第 6 个配体。可写 `[0, 1, 2]`，也可写 `"all"` 选择全部。
- 总数 = 选中的配体数 × `SAMPLES_PER_INPUT`。例如 `[5]` 和 `5` 共生成 5 个；选 10 个配体、每个生成 10 个就是 100 个。
- `SEED_START`：起始随机种子；相同输入与种子用于复现，换一段种子用于再次采样。
- 相对路径从项目根目录计算；支持包含空格的路径以及 Linux 绝对路径，也支持仓库外的数据和结果目录。
- 每次新实验用新目录，例如将三个输出相关路径中的 `experiment_001` 一起改为 `experiment_002`；不会自动覆盖或把两批混在一起。
- `INPUTS_PER_GROUP` 保持 2 即可。组大小不改变判别标准，同一输入的全部种子保持在同组。
- 输入检查发现未满足现有 scaffold hopping 前提的配体，会在生成前列出并停止；不会偷偷换一个配体凑数。不扫描、排除旧输入，也不读取旧实验日志。

## 方式一：一次完成全部流程

确保 `INTERVENTION_INPUT_DIR` 正好是本次 `GENERATION_DIR` 的 `failed` 子目录。

```bash
python scaffold_validation100/run.py --repo .
```

依次生成、判别、定位、分类，再对本次失败样本运行现有干预及对照。

## 方式二：分两步运行

先生成、判别和定位，保存成功/失败目录：

```bash
python scaffold_validation100/run.py --repo . --generate-only
```

然后确认 `INTERVENTION_INPUT_DIR` 指向要处理的失败目录，并给 `INTERVENTION_OUTPUT_DIR` 填一个新的输出目录：

```bash
python scaffold_validation100/run.py --repo . --intervene-only
```

第二条命令**不生成新分子**。它读取所选失败样本，并自动保留同一输入配体的成功样本作为对照。对照不计入修复成功数。

如只想处理部分失败样本，把 `failed/` 下选中的**整个样本子目录**复制到一个新文件夹，再把 `INTERVENTION_INPUT_DIR` 指向这个文件夹。每次选择同一批生成结果中的样本。不要只复制 SDF、修改 SDF、删除 `sample.json`，也不要混入另一个实验的样本。

若全部样本原本成功，失败目录为空也能正常完成，报告请求干预 0 个。

## 输出在哪里

| 路径（以上面配置为例） | 内容 |
|---|---|
| `outputs/experiment_001/generated/success/` | 原始生成即通过完整判别的样本 |
| `outputs/experiment_001/generated/failed/` | 原始生成未通过的样本 |
| `outputs/experiment_001/generated/group_*/` | 原始生成轨迹、判别和定位证据 |
| `outputs/experiment_001/generated/inputs/` | 本次 SDF/PDB 的内容副本，避免后来更换原文件影响旧实验 |
| `outputs/experiment_001/intervention/success/` | 请求干预后修复成功的样本 |
| `outputs/experiment_001/intervention/failed/` | 请求干预后仍未修复的样本；`result.json` 区分未通过、不支持与分支错误 |
| `outputs/experiment_001/intervention/group_*/rescue_v2/` | 所有干预和对照的详细结果 |
| `outputs/experiment_001/intervention/summary.json` | 请求数量、修复数量、未修复数量、分支错误等 |

每个原始样本子目录里有 `molecule.sdf` 和 `sample.json`；后者记录输入、原始记录位置和失败判据。若生成过程无法解码出分子，SDF 可能为空，这个失败样本依然计入统计，不会被丢弃。

每个干预结果子目录里有 `molecule.sdf` 和 `result.json`。成功目录保存通过判别的干预分子；失败目录保留原始失败分子，各尝试分支在详细结果目录内。原始 `generated/` 不会因为修复成功而移动或改写分子。

**保留整个 `generated/` 目录。** 后续定位和干预需要原始张量、原子映射、判别报告和轨迹，不能只留下分类后的 SDF。目录中的引用记录不等于已经复制所有轨迹。

修复统计仍区分不移动固定坐标的 A/B/AB 与允许移动固定坐标的 C。C 不是相同固定坐标条件下的成功；随机/力场对照不能冒充修复结果。

## 发回什么

完整运行或单独干预结束，发终端提示的：

```text
outputs/experiment_001/intervention/validation100_review.zip
```

只运行生成步骤时，发：

```text
outputs/experiment_001/generated/validation100_review.zip
```

名称保留兼容习惯，实际数量不限制 100。包中包含本次输入副本、分类和详细结果，省略大型 `.pt` 和轨迹 `.xyz`；这些仍需留在你的机器上。

## 准备、继续与回退

只检查输入并固定本次计划：`python scaffold_validation100/run.py --repo . --prepare`。

运行该计划：`python scaffold_validation100/run.py --repo . --run --output outputs/experiment_001/generated`。此命令只完成生成、判别、定位与分类；随后用 `--intervene-only` 做干预。已完成且完整的步骤会复用；已失败或输出被修改的步骤不会自动重复，以免混合结果。改 `config.py` 不会改变已固定的计划。

安装后不能用新代码续跑旧版本的半成品实验。先结束运行再更新；旧结果和备份均保留。

需要回退时，使用安装时输出的实际备份路径：

```bash
python scaffold_path_update/install.py --repo . --restore archive/validation_paths_backup_实际编号
```

## 检验范围

- CPU 软件测试覆盖路径、输入数量、完整运行、分步运行、失败子集、空失败目录、覆盖防护、文件变更检查、安装及回退。
- 用前两批已评估的 200 个分子记录复核分类和统计，保持 66 失败/59 修复以及 55 失败/50 修复；没有重算这些分子的化学评价或干预。
- 现有 29 项定位软件测试通过。生成参数列表、非入口评价函数、定位与修复公式保持原样。
- 本环境没有执行真实 FLOWR GPU 生成或重新执行分子干预。安装后先用默认 5 个样本做本地验证，再扩大数量。输入路径支持扩展不代表已验证所有数据集或其他模型。
