# 论文结论 ↔ 数据文件 ↔ 复核结论（骨架，待完整矩阵后填写）

| 论文结论 | 数据文件 / 脚本 | 复核结论 |
|---|---|---|
| FSHGRL 在 15 个配置上的达成率高于 NF 与 NG（消融表、配对差值图） | `result/ablation_results.csv`、`advantage_table.csv`（run_09） | 待跑：完整矩阵未运行 |
| FSHGRL 在全部配置上高于 MOR/FIFO/MWKR/Random/RRC（规则对比表） | `result/eval_results.csv`、`rules.csv`、`paper_assets/tables/comparison_pdr.tex` | 待跑；规则为新实现（known-issues 第 4 条） |
| FSHGRL 高于 DRL 对比方法（DRL 对比表） | `paper_assets/tables/comparison_drl.tex` | 待跑；本仓库基线 A2C/DQN/DDQN 为新实现，不是论文的 DRLG/AHP-DQN/HSDDQN（known-issues 第 3 条） |
| 配对 Wilcoxon：五个对比的 p_Holm = 3.052e-4 | `result/paired_tests.csv`；`tests/test_stats.py` | 检验实现已按论文描述复算出论文表中的 R+、p、p_Holm（给定全胜/一平的差值模式） |
| 决策时间随 DDT 与 S 的变化 | `result/eval_results.csv` 的 `decision_time_ms`（run_07）、决策时间图 | 待跑；口径与平台差异见 known-issues 第 5 条 |
| 训练曲线：早期改善、后期波动下降 | `result/ours/<id>/run<k>/curve.csv`、训练曲线图 | 待跑；原成品图不可再生成（known-issues 第 15 条） |
| 工业 3D 打印案例 | 无 | 不可复现（known-issues 第 14 条） |
| 流体引导剪枝候选动作 | `environment/simulator.py` | **代码与描述不符**：训练器不剪枝，流体只作为特征（known-issues 第 2 条） |
| 超参表（1000 epoch、熵系数 0） | `configs/algo.yaml` | **与代码不符**（2000、0.01；known-issues 第 1 条） |
