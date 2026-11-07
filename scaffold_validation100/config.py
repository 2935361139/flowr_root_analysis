"""日常只改这一页。相对路径从项目根目录算起，也可填写 Linux 绝对路径。"""

# 1. 输入：SDF 中的配体必须与这个蛋白 PDB 使用同一坐标系。
SDF_PATH = "examples/cdk2_ligands.sdf"
PDB_PATH = "examples/cdk2_protein.pdb"

# 选 SDF 中哪些配体：编号从 0 开始；[5] 是第 6 个；"all" 是全部。
LIGAND_INDICES = [5]
# 总生成数量 = 选中的配体数 × 每个配体生成数量。这里共生成 5 个。
SAMPLES_PER_INPUT = 3
SEED_START = 3001

# 2. 原始生成结果：自动分到 success/ 和 failed/，轨迹也保留在此目录。
# 每次新实验用一个新目录名，例如 experiment_002；不会覆盖已有结果。
GENERATION_DIR = "outputs/experiment_002/generated"

# 3. 待干预的失败目录：完整流程填上面目录的 failed 子目录。
# 单独干预时，可指向另一批生成的 failed/，或复制了部分样本子目录的新目录。
INTERVENTION_INPUT_DIR = "outputs/experiment_002/generated/failed"

# 4. 干预结果：success/ 是修复成功；failed/ 是尚未修复（含不支持、分支出错）。
INTERVENTION_OUTPUT_DIR = "outputs/experiment_002/intervention"

# 通常不需要改：每组包含几个输入配体；不改变总数量和判别标准。
INPUTS_PER_GROUP = 2

#运行全部流程                                python scaffold_validation100/run.py --repo .
#先生成、检查、定位和分类                     python scaffold_validation100/run.py --repo . --generate-only
#再读取指定的失败目录做干预，不重新生成        python scaffold_validation100/run.py --repo . --intervene-only
