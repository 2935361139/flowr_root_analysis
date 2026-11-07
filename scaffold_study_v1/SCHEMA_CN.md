# 共用接口 scaffold-evidence/1

顶层：`schema`、`model`、`runs`、`provenance`。
每个 run 包含唯一 `run_id`、`input_id`、`input_group`、`seed`、`endpoints`、`observations`。
输入组由适配器按参考配体和环境身份建立，不能把随机种子当作输入组。
实验拆分还需要在这些组之上按相关骨架/口袋聚类，本版不自动创建独立测试集。

`endpoints.raw` / `endpoints.decoded` 都包含以下 bool 或 null 字段：

```text
chemical_valid
connected
fixed_chemistry_preserved
fixed_substructure_match
geometry_valid
scaffold_changed
```

raw 在本 FLOWR 适配器中表示“未经价态修复的终态解码”，不冒充解码前的神经网络张量。
connected 的 raw 字段使用已评价原始图；decoded 只在化学有效时使用原生连通检查，无效解码保留未知。
其他模型若没有两个解码阶段，需要明确提供阶段语义及未知值，不能复制字段假装做过两次检查。

每条 `observations` 包含 `stream`、`entity`、`metric`、`points`。
终态可选列表 `fixed_changed_atoms`、`neutral_carbon_overvalent_atoms` 明确相关固定原子。
对应 entity 使用 `fixed_atom:原子编号`。只有终态与时间线指向同一个固定原子，才关联该时间证据；不能用其他原子的异常解释当前终态失败。
支持的 metric 为 `fixed_chemistry_changed`、`neutral_carbon_overvalence`、`disconnected`。
无对应观测的 metric 不生成序列。其他失败类型仍可通过终态标签保留，不能自动声称时间定位。

points 每条包含：

```json
{"index": 0, "native_time": 0.0, "stage": "native_stage_name", "value": null}
```

index 是该条独立记录流的连续零基编号；native_time 必须保留模型的原生语义和方向，不转成跨模型同一物理时刻。
value 的 null 表示无法判定，不得转成 false。最后一段连续 true 的起点只报告为持续观察区间。
第二模型适配器若有不连续记录，应显式补入未知占位并标记阶段，不能跳过缺帧伪装连续。

可选 `geometry.raw/decoded` 含 `clash_pairs` 和 `reference_context`，每个碰撞记录至少含 `check` 和本模型内明确的原子标识。
参考同名失败与同一原子对原本碰撞必须分开；当前生成原子不能用同号直接对应参考骨架。

输出 `scaffold-diagnosis/1` 始终把 `causal_status` 保持为 `not_tested`，内部模块为 null。
未来干预结果应另建带实际实验记录、对照和结果校验的证据格式；禁止手改这个标签把观察升级为因果证据。
