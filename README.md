# SCB-SAM 主表与消融复现项目

**当前状态：源码预发布。主表配置、历史结果与研究源码可供核查；新机器完整训练复现尚未验收。**

本次按作者确认先公开源码，继续补齐运行验证。`reproduce.py train` 尚未接通便携训练，不能将本预发布当作一键复现版本。173份上游源码通过固定版本和哈希在仓库外获取；另有一份SDCL二分类适配文件暂不公开，具体范围见[外部依赖说明](docs/EXTERNAL_SOURCES.md)。

范围是论文当前主表28个实验，以及H、HP、HQ、HPQ四组配置。没有额外with-SAM比较、已删除的开发候选或新的训练结果。统一入口为`reproduce.py`；实际模型、训练器与各方法依赖保留为独立模块。

## 当前可以运行

以下命令仅读取已归档结果、配置和代码状态，不启动训练或重新评价医学影像。Python 3.10及以上可运行这些整理工具。

```bash
python reproduce.py list
python reproduce.py report --experiment BY_scb_seg_n005
python reproduce.py report --experiment ablation_HQ
python reproduce.py plan --experiment ablation_HPQ
python reproduce.py verify
```

`verify`在仍有发布阻碍时明确返回退出码2和`NOT_READY_TO_RELEASE`，即使目录结构检查通过也不会宣称训练复现通过。`train`目前明确拒绝派发并列出阻碍；它不是已经实现的便携训练接口。

## 项目内容

| 内容 | 位置 |
| --- | --- |
| 统一检索、结果报告、计划与状态检查入口 | `reproduce.py` |
| 主表28行、冻结配置与来源哈希 | `records/main_table.json`、`records/configs/` |
| 消融H/HP/HQ/HPQ配置 | `records/configurations/` |
| 四组无注释训练脚本及原依赖研究快照 | `ours/` |
| 比较方法原实现的脱敏研究快照 | `baselines/source_snapshots/` |
| 每项实验的历史入口及移植状态 | `scripts/experiment_routes.json` |
| 初始化的来源、原始SHA与是否可选 | `checkpoints/manifest.json` |
| 固定病例划分、顺序与原始SHA核验 | `splits/` |
| 数据与预算说明、筛选历史、当前阻碍 | `docs/` |
| 历史环境信息 | `environments/` |

`ours/hp_hq_final.py`支持四组独立路由。H/HP/HPQ保持原科学训练循环，HQ固定CCBS系数0.25。已完成语法、无注释/无文档字符串、只读结果和隔离路由检查；未进行新GPU训练。它仍依赖原环境的准备与数据身份回执，不等同于在新机器可直接训练。

## 代码与权重分离

源码项目不包含`.pth`等大文件。复现时按方法获取对应SAM预训练，而不是每项实验下载全部权重。BY5/BY10/Public5/Public10监督阶段权重可作为便利下载；若提供完整监督阶段生成流程，也可以先生成再接续。H/HP/HQ/HPQ必须复用同一次BY5监督起点。

此前整理时取得的权重仅保存在源码仓库外的本地资产目录，尚未公开上传。上游精确下载地址与版本需在发布前固定并校验；本项目没有虚构可用下载URL。详情见[权重说明](checkpoints/README.md)。

Public433是IDEAS中433例带切除区掩码的术前T1子集。影像和掩码有不同下载入口，见[数据获取说明](docs/historical_data.md)。四组历史划分已核验，公开下载病例到内部别名的对应关系未核验；不能承诺仅下载数据后即可原样复现实验。

## 必须如实保留的结果边界

- 旧合并脚本的HQ=0.25末轮结果真实存在，但当时未保存对应模型权重，不能提供该历史最终模型下载。
- HQ是已见开发结果后的系数选择，不把与冻结HPQ的差值包装成匹配消融的有效性证明。
- Public报告终点及选择过程与BY固定末轮不同，按每行配置和来源记录解释，不能统一改写成固定同轮。
- 研究快照包含历史路径/契约占位符。直接替换路径、删掉SHA检查，不能视为完成移植。

具体见[主表来源](docs/MAIN_TABLE.md)、[消融筛选](docs/ABLATION_SELECTION.md)及[当前发布状态](docs/READINESS.json)。

## 底层源码核验完成

三个此前缺少可读源码的级联模块已补入`ours/research_snapshot/segment_anything`和比较方法共用快照。两份来自历史源码候选；`build_sam3D_cascade.py`的准确历史原文件未找到，本次依据实际训练字节码恢复。三份发布源码重新编译后，仅归一文件路径，完整代码对象与训练版本一致，包括嵌套函数和行号信息。恢复身份、SHA及校验方法见[源码核验说明](docs/CASCADE_SOURCE_RECOVERY.md)。这一检查不等同于完整新机器训练验收。

## 剩余阻碍

1. 原数据身份、科学源码SHA、共同初始化与准备回执依赖历史环境。需要完成新环境准备流程和等价性检查，同时保留原保护；历史`.pyc`散列记录不会被伪造为新增`.py`散列。
2. 完整依赖环境、监督前缀生成、训练/推理/评价入口尚未通过干净环境验收。
3. 第三方来源与再分发条件需按逐组件记录完成核对。

## 许可

Apache-2.0适用于项目自有代码。第三方来源不因这一选择自动改用Apache-2.0；应保留其许可证与引用。详细边界见`docs/THIRD_PARTY.md`。权重未随源码发布。
