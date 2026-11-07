"""日常只改这一页。这里只填写常量，不写函数或导入语句。

输入格式：(蛋白示例名称, 配体在对应 SDF 中的编号)，编号从 0 开始。
目前使用现有 recorder 支持的 cdk2 / tyk2 / ptp1b / bace 示例。
默认为上次成功完成的 10 个输入；允许重复实验，不读取任何旧 outputs。
相同输入和种子用于复现；更换 SEED_START 用于再次采样。
"""

# 增删下面的条目，就能改变输入配体数量。
INPUTS = [
    ("cdk2", 5),
    ("tyk2", 4),
    ("ptp1b", 4),
    ("bace", 3),
    ("cdk2", 6),
    ("tyk2", 5),
    ("ptp1b", 5),
    ("bace", 4),
    ("cdk2", 8),
    ("tyk2", 6),
]

# 每个输入生成多少个。总数 = len(INPUTS) × SAMPLES_PER_INPUT。
SAMPLES_PER_INPUT = 10

# 每个输入使用 SEED_START 开始的连续种子。
SEED_START = 2001

# 每组包含几个输入；同一输入的所有种子保持在一组。
# 每组样本数自动计算，不再写死为 20。最后一组可以更小。
INPUTS_PER_GROUP = 2

# 自动在 outputs 下创建新目录，名称由此前缀和时间组成。
OUTPUT_PREFIX = "sh_validation"
