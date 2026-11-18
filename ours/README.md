# 我们方法与四组消融

本组件只保留 H、HP、HQ、HPQ 四组，不包含被用户取消的最新实验，也不包含四项关键设计筛选。统一入口是 `hp_hq_final.py`；底层科研代码位于 `research_snapshot`，这些模块是训练依赖，不是新增实验。

## 可用范围

目前可直接执行 `report` 查看脱敏后的真实历史结果。四组训练路由、语法和注释检查通过，训练步骤、模型构建及损失函数的 AST 保持原实现。该导出不是已验证的新机器训练环境：原脚本严格绑定监督初始化验收、代码散列、数据身份和准备回执。改变根目录后必须在目标机器合法重新建立完整准备与验收链；本次没有把校验删除、伪造为已通过，也没有重新训练。

```bash
python ours/hp_hq_final.py --action report --group all
python ours/hp_hq_final.py --action report --group HQ
```

训练路由如下：H、HP、HPQ使用原四组训练循环 `run_hp`（函数名来自早先合并版本，实际接受三组）；HQ固定CCBS=0.25，使用 `configure_ccbs` 和 `run_hq`。`both`只表示HP/HQ，`all`表示四组。所有正式训练从共同初始化完成36轮、684次成功更新、3420对主输入；桥接状态不接入正式训练。

目标机器环境配置必须显式指向完整、已验收的项目：

```bash
export ASNET_ROOT=/absolute/path/to/validated/project
export ASNET_GPU_UUID=your-device-uuid
python ours/hp_hq_final.py --action preflight --group H --run-id h_reproduce_01
python ours/hp_hq_final.py --action preflight --group HP --run-id hp_reproduce_01
python ours/hp_hq_final.py --action preflight --group HQ --run-id hq_reproduce_01
python ours/hp_hq_final.py --action preflight --group HPQ --run-id hpq_reproduce_01
```

预检通过且研究者明确准备运行后，对指定组使用同样参数把 `preflight` 改成 `train`。入口先运行有限桥接，然后从共同初始化重新开始正式训练。`worker`为内部入口，不独立手动启动。不得在原已存在目录重复派发，也不要把示例当成已完成GPU验证。默认未指定ASNET_ROOT时仅取当前工作目录；这不代表当前目录已具备训练条件。

## 初始化权重

本组件列明两份必要初始化，具体大权重由项目级权重目录统一提供，未重复复制进ours：

| 权重 | 用途 | SHA256 |
| --- | --- | --- |
| SAM-Med3D turbo | 构建模型所需预训练状态 | `899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c` |
| BY5共同监督初始化 | 60轮、300次监督更新的固定起点，四组共用 | `b113d1731a8e535ff482bebe7c2ca0d821760d44e5ea02e59b33cae58557c4be` |

预期相对路径记录于 `dependency_inventory.json`。SAM初始权重的再分发遵循其上游许可；监督初始化是研究数据训练产物，其可公开性也须按数据和模型权重条款确认。本组件不把第三方权重自动许可为Apache-2.0。

## 仍需完成的新机器绑定

1. 底层目录布局要包含 `expes/semi05_gate/by5_four_arm_training_20261030`、监督初始化训练和验收依赖，以及模型/数据加载模块；85份Python快照已列在散列清单。
2. `four_contract.verify_prepared`仍要求原有BUNDLE/PREPARE绑定、监督初始化验收、六根初始状态、原数据划分与文件字节验证。公开包未带有私有病例编号、原inode/mtime或机器历史状态，不能仅凭复制本目录跳过这些条件。
3. `source_manifest.json`、资格验收manifest、ancestry/acceptance、preparation等状态依赖不是可机械沿用的跨机器文件。路径与源码散列改变后需要新版本的准备/验收流程，同时验证科学函数与原版一致。现有旧散列保留为来源证据，不声称新源码仍拥有旧散列。
4. 保留Linux `fcntl`、`/proc`、原GPU设备FD/独占锁、内存/磁盘容量检查和原子保存读回。仅将私有根和设备UUID变为环境变量，不意味已解除原物理GPU0检查。
5. 表中评价采用GT ROI、历史开发fold2、单个seed2027和voxel距离单位，不是独立测试或毫米距离。

`source_transform_manifest.json`给出原整理文件的原SHA、新SHA、AST哈希、发生变化的函数及允许的路径/设备迁移。统一训练入口仍无注释和docstring，SHA保持不变。后续补入的三个底层模型模块去除普通注释，但保留作为运行时属性的原文档字符串，以保持完整编译代码对象一致；解释及原注释另见项目`docs/CASCADE_SOURCE_RECOVERY.md`。版权注记见 `SNAPSHOT_NOTICES.md`，不能因此移除第三方许可要求。H/HPQ原真实历史入口也保存在 `research_snapshot/expes/semi05_gate/by5_four_arm_training_20261030/four_control.py` 和 `four_training.py`。统一路由没有调用旧四组队列或重新运行已经结束的实验。

结果筛选与HQ权重缺失情况见项目 `docs/ABLATION_SELECTION.md`。
