# 实验设计规格（规格化摄取自论文与原仓库）

## 0. 问题、范式与声明

- 问题：DFFSP-HFOI，最大化订单按时达成率 `eta`；契约目标值 `obj = 1 - eta`（最小化）。
- 训练范式：**逐算例型（per_instance）**。训练集 = 评测集 = 15 个基准算例；每个 (算例, run) 一个训练作业；
  论文结论不涉及跨算例泛化。
- 取点协议：每轮 PPO 更新后，当前策略（`policy_old`，已与 `policy` 同步）在该算例上跑一次贪心 episode，
  `obj` 进 `curve.csv`；`obj_best` = 历史贪心最优与训练期采样 episode 最优的较小者。
- 模型选择：`best_greedy_checkpoint.pt`（贪心目标值最优轮）；原仓库口径 `best_checkpoint.pt` 并列保留。

## 1. 算例

| 档位 | 算例 | 来源 | 用途 |
|---|---|---|---|
| benchmark | DDT ∈ {500, 600, 900, 1200, 1600} × S ∈ {50, 100, 150}，共 15 个 | 外部给定（原仓库 `data/instance/competition/`） | 训练 + 评测（全部论文表格） |
| synthetic | `data/generator.py` 按论文参数表生成 | 新 | 不进论文结果 |

算例参数表（论文）：5 产品 × 5 工序 × 每工序 5 机；S ∈ [20, 200]；加工时间 rand[25, 450] s；到达间隔 rand[1, 200] s；
DDT rand[300, 2500] s。基准文件与该表的差异见 `docs/known-issues.md` 第 7 条。

## 2. 方法与 run 矩阵（`docs/run-matrix.md`）

| 方法 id | run 数（`configs/runs.yaml`） | 配置 |
|---|---|---|
| ours | 3 | `configs/algo.yaml` |
| nf、ng | 各 3 | `configs/ablation/{nf,ng}.yaml` |
| a2c、dqn、ddqn | 各 1（固定 seed 1） | `configs/baselines/*.yaml`，预算同 ours |
| fifo、spt、mor、mwkr | 1（确定性） | `baselines/run_rules.py` |
| random、rrc | 10 | 同上 |

## 3. 训练参数表 v1（`train_confirmed: true`，来源：原仓库 `data/config.json`）

| 参数 | 值 | 论文超参表 |
|---|---|---|
| 迭代（epoch）数 `max_iterations` | 2000 | 1000 |
| 学习率 `lr` | 3e-4，`lr_schedule: none` | 3e-4 |
| 折扣 `gamma` / GAE `gae_lambda` | 1.0 / 0.95 | 1.0 / — |
| PPO 轮次 `K_epochs` / 截断 `eps_clip` | 3 / 0.2 | 3 / 0.2 |
| 损失系数 `A_coeff` / `vf_coeff` / `entropy_coeff` | 1.0 / 0.5 / 0.01 | 1.0 / 0.5 / 0 |
| `max_grad_norm` / `target_kl` | 0.5 / 0.03 | — |
| 每轮 episode 数 `num_envs` / `minibatch_size` | 8 / 512 | — |
| 候选订单 `order_top_k` / 流体 `fluid_mode` | 5 / cached_lp | 流体引导 |
| 网络：嵌入 16/16，隐藏 256，actor/critic 128 × 3 层，候选自注意力 1 头 | `configs/algo.yaml` | — |
| 优化器 | Adam (0.9, 0.999) | Adam |

v2（预实验校准）未进行：预实验关闭（见变更记录）。

## 4. 评测

`eval.py`：贪心 1 次 + 采样 10 次（逐条独立，`batch_episodes: 1`），决策时间分项（策略前向、观测构建、流体、仿真推进），
CPU 单线程以保证各方法可比；规则基线同一仿真器、同一候选集。

## 5. 统计

每算例取 run 均值；主方法对每个对比方法做配对 Wilcoxon 符号秩检验（精确枚举、零差剔除、平均秩、两侧），
Holm 校正（家族 = 全部对比方法）；Friedman 检验给平均秩；均值 ± 标准差（单 run 留空）。

## 6. 落盘契约（6B）

数据侧：`data/instances/index.csv`、`<id>/params.yaml`、`reference.csv`。
run 侧：`train_metrics.csv`（原 49 列）、`curve.csv`（`epoch, instance_id, obj, obj_best, gap, feasible, wall_time_s`）、
`log.csv`（`epoch, env_steps, wall_time_s, train_obj_mean, fixed_obj_mean, fixed_gap_mean, policy_loss, value_loss,
entropy, clip_frac, approx_kl, lr, sps, gpu_util, gpu_mem_gb, train_fulfillment_mean, fixed_fulfillment_mean,
eval_seconds`）、`fixed_set_final.csv`、`config_snapshot.yaml`、`train_meta.yaml`、检查点。
汇总侧：`eval_results.csv`、`incumbents.csv`、`rules.csv`、`stats_summary.csv`、`paired_tests.csv`、
`ablation_results.csv`、`advantage_table.csv`、`friedman.csv`、`conv_<method>.csv`。
`generalization.csv`、`sensitivity.csv` 不在本设计中（无跨规模泛化与敏感性实验）。

## 7. 变更记录

| 日期 | 变更 | 理由 |
|---|---|---|
| 重构 C9 | 预实验关闭（`configs/pilot.yaml` `enabled: false`） | 逐算例范式下训练集即评测集；pilot 档会把 CPU 预算花在不进结果的算例上 |
| 重构 C4 | `curve.csv` 每轮一次贪心 episode（`eval_every: 1`，约 +5% 墙钟） | 技能的取点协议；回归测试证明不改变训练数值 |
| 重构 C3 | 训练参数表 v1 = 原仓库配置值 | 保持数值行为；与论文超参表的差异记入 known-issues |
