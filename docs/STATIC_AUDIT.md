# 发布前静态审计

仅审计当前公开项目副本，不联网、不导入项目代码、不启动训练、不读取权重。所有输出只保留文件路径、行号、分类与计数，不记录敏感原值。

共扫描675个文件，601个Python文件全部AST解析通过，失败0。私钥、凭据赋值、常见令牌、私网IP和项目病例编号模式均无命中；这不等于对任意秘密格式的完备保证。

## 已修复

`baselines/source_snapshots/SDCL-master/code/dataloaders/acdc_data_processing.py` 中2处上游已掩码home路径改为显式环境配置，注释保留，原/新SHA及AST变化见 `STATIC_AUDIT_REMEDIATION.json`。原历史来源未改。它们不是已确认的真实用户名泄露，但原写法无法跨机器直接使用。

`baselines/original_source_map.json` 的497条旧发布前缀已修正；496条当前文件SHA验证通过，7条旧VNet监督参考标记为不在当前范围；非预期缺失0、散列失配0。见 `SOURCE_MAP_REPAIR.json`。

## 尚未完成的可移植性工作

数据/环境路径文本候选仍有42处、涉及29个文件，其中包含拼接到数据根的路径后缀和上游默认示例，不可把全部命中等同于私有绝对路径。按AST分出的分类与位置见JSON；对应实际入口需要逐项确认数据根和输出根，不能声称已portable。

我们的严格训练准备链有8项原有manifest/初始化验收/准备元数据未纳入公开快照。它们含机器或数据绑定，不能直接复制私有版本或去掉校验；目标机器需要重新生成并核验。已有BUNDLE/PREPARE等检查仍在，静态语法通过并不意味着训练预检通过。本文件的计数是原审计快照；后来补齐的三个底层模块及其完整字节码等价验证另见`CASCADE_SOURCE_VERIFICATION.json`，不得把该验证等同于新机训练通过。

### 需要目标机重新准备的元数据位置

- `ours/research_snapshot/expes/semi05_gate/by5_four_arm_training_20261030/manifest.json`
- `ours/research_snapshot/expes/semi05_gate/by5_four_arm_initialization_check_20261030/manifest.json`
- `ours/research_snapshot/expes/semi05_gate/by5_four_arm_initialization_check_20261030/ancestry.json`
- `ours/research_snapshot/expes/semi05_gate/by5_four_arm_initialization_check_20261030/acceptance.json`
- `ours/research_snapshot/expes/semi05_gate/by5_four_arm_rerun_20261030/source_manifest.json`
- `ours/research_snapshot/expes/semi05_gate/by5_four_arm_rerun_20261030/dependency_pins.json`
- `ours/research_snapshot/expes/semi05_gate/by5_four_arm_rerun_20261030/infrastructure_pins.json`
- `ours/research_snapshot/expes/work_dir/semi05_gate/BY5_FOUR_ARM_RERUN_20261030/control/preparation.json`

### 路径候选文件

- `scripts/baseline_required_directories.json`：1处（行 61）
- `ours/research_snapshot/utils/data_loader.py`：1处（行 184）
- `baselines/source_snapshots/utils/data_loader.py`：2处（行 294,308）
- `baselines/source_snapshots/SemiSAM-main/code_semisam+/networks/net_factory.py`：1处（行 13）
- `baselines/source_snapshots/SDCL-master/code/test_ACDC.py`：2处（行 37,81）
- `baselines/source_snapshots/SDCL-master/code/test_LA.py`：1处（行 36）
- `baselines/source_snapshots/SDCL-master/code/dataloaders/dataset.py`：5处（行 40,42,95,97,99）
- `baselines/source_snapshots/SDCL-master/code/dataloaders/la_heart_processing.py`：1处（行 10）
- `baselines/source_snapshots/SDCL-master/code/pancreas/test_util.py`：4处（行 295,297,313,315）
- `baselines/source_snapshots/SDCL-master/code/utils/test_3d_patch.py`：4处（行 24,120,283,285）
- `baselines/source_snapshots/expes/semi05_gate/main_table_20261019_r4/sfr_runtime_components.py`：1处（行 5）
- `baselines/source_snapshots/expes/semi05_gate/public_hpq_epochdiag_20261019/hpq_initialization.py`：1处（行 99）
- `baselines/source_snapshots/expes/semi05_gate/public_hpq_fineq_n010_restart4_20261027/hpq_initialization.py`：1处（行 99）
- `baselines/source_snapshots/expes/semi05_gate/public_hpq_fineq_r3_20261020/hpq_initialization.py`：1处（行 99）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/SFR/code/test_singleclass.py`：1处（行 11）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/SFR/code/train_finetuning.py`：1处（行 16）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/SFR/code/train_retraining.py`：1处（行 16）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/ReLiF_3D_train_brats.py`：1处（行 43）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/ReLiF_3D_train_la.py`：1处（行 46）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/test_3D_brats.py`：1处（行 13）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/test_3D_la.py`：1处（行 14）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/test_3D_util_brats.py`：1处（行 93）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/test_3D_util_la.py`：1处（行 93）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/val_3D_brats.py`：1处（行 94）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/val_3D_la.py`：1处（行 94）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/dataloaders/brats2019.py`：1处（行 39）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/dataloaders/la.py`：1处（行 39）
- `baselines/source_snapshots/expes/semi05_gate/main_table_cpu_candidates_20261019_1822/official/ReLiF_3D/code/networks/net_factory.py`：1处（行 13）
- `baselines/source_snapshots/expes/semi05_gate/dualteacher_fixed_final_20261021_v4_conda/native/utilsm/translate_labels.py`：2处（行 28,29）
