# Run 矩阵与代价

| 脚本 | 作业数 | 单作业（本机 4 核 CPU，8 环境，2000 迭代） | 合计 |
|---|---|---|---|
| run_03 ours | 15 算例 × 3 run = 45 | 12 h（S=50）～48 h（S=150） | 约 1350 CPU·h |
| run_04 nf、ng | 2 × 15 × 3 = 90 | 同上 | 约 2700 CPU·h |
| run_05 a2c、dqn、ddqn | 3 × 15 × 1 = 45 | PPO 量级（DQN/DDQN 每轮另加 50 次 TD 更新） | 约 1500 CPU·h |
| run_06 规则 | 1 | 1 min | 1 min |
| run_07 评测 | 180 个检查点 | 约 30 s | 1.5 h |
| run_09 聚合 | 1 | 1 min | 1 min |

本次重构在本机只验证了冒烟（2 迭代）与小规模矩阵机制（4 个微型作业 + 评测 + 聚合），完整矩阵未运行；
`result/` 下没有论文级数据。减小预算的两种方式：`configs/runs.yaml` 降低 run 数；`configs/algo.yaml` 的
`max_iterations`（论文超参表为 1000）。
