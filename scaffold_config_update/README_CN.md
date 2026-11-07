# 统一配置入口更新包

本包只替换 `scaffold_validation100/` 以及 `scaffold_study_v1/terminal_control.py`。不修改生成器、模型权重、原评价、定位、几何修复算法和实验数据。

## 安装一次

先等当前实验结束。将 ZIP 解压到项目根目录，确认有：

`/home/ljl/projects/flowr_root/scaffold_config_update/install.py`

在 WSL 终端运行：

```bash
cd /home/ljl/projects/flowr_root
source .venv/bin/activate
python scaffold_config_update/install.py --repo .
```

安装前验证当前方法文件与已核对版本相符，并备份旧的整个 scaffold_validation100 文件夹和 terminal_control.py 到 archive/validation_config_backup_随机后缀/。若版本不符或仍有 RUNNING.lock，将在替换前停止；请发报错，不要绕过。安装中写入失败会恢复旧文件。重复安装会保留你的 config.py。

不要再执行旧的 install_compat.py，也不必手动删除旧入口文件。它们已随原文件夹备份。历史 outputs 不会被安装器修改。

## 日常操作

打开 `scaffold_validation100/config.py`，按需要修改输入清单、SAMPLES_PER_INPUT、SEED_START、INPUTS_PER_GROUP。

然后执行：

```bash
python scaffold_validation100/run.py --repo .
```

每次自动新建输出目录，允许重复输入，不扫描旧实验。每批只进行配置中计划的生成次数和原定干预/对照，不因失败自动加样。

详细参数说明见安装后的 `scaffold_validation100/README_CN.md`。

## 回到更新前版本

先结束正在运行的实验，然后用安装时打印的实际备份目录替换下方路径：

```bash
python scaffold_config_update/install.py --repo . --restore archive/validation_config_backup_实际后缀
```

恢复前会另外保存当前新版代码和 config.py。原始实验数据不受影响。恢复后若想重新使用新版，再运行安装命令。

## 验证范围

见 VALIDATION.json 和 software_tests.log。CPU 调度测试采用合成子进程，不产生真实分子。生成、判别和干预核心版本对比，以及已有诊断测试另行记录。本环境未执行完整 GPU 分子生成或 PoseBusters 复评，不能承诺新一批结果数字完全相同。
