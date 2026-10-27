# 预实验日志

预实验关闭（`configs/pilot.yaml` `enabled: false`），未进行任何轮次。

理由：逐算例范式下训练集即评测集，没有"固定评测集上先验证收敛再投算力"的位置；一个独立的 pilot 档会把 CPU 预算
花在不进任何结果的算例上。若需启用：`enabled: true`，`instances` 取每个 DDT 档一个代表算例，
`python scripts/run_02_pilot.py` 训练 `methods` 并写 `result/pilot/pilot-report.md` 与 `verdict.txt`
（主方法在多数 pilot 算例上达到最优规则参考值即 PASS）。每轮改动请在此记录，最多 3 轮。
