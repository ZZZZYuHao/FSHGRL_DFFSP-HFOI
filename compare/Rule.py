"""
对比实验：与四种启发式调度规则的比较
四种启发式调度规则：
- MOR  : Most Operations Remaining（选择剩余工序最多的工件）
- FIFO : First-In First-Out（选择先到达的工件）
- MWKR : Most Work Remaining（选择剩余总处理时间最多的工件，按平均加工时间估算）
- SPT  : Shortest Processing Time（选择处理时间最短的工件）
"""
import numpy as np
import torch
import random


class RulePolicy:
    def __init__(self, rule: str, device: str = "cpu"):
        rule = rule.lower()
        assert rule in {"mor", "fifo", "mwkr", "spt", "rand"}, f"未知规则: {rule}"
        self.rule = rule
        self.device = device

    def _collect_candidates(self, state):
        """
        返回候选列表：
        [(action1, action2, (j, m, r), feature_dict), ...]
        action1: 对应 state.action_dict 的键（映射到 (j, m)）
        action2: 该工序 (r,j) 队列中的订单索引
        (j, m, r): 方便取特征的索引组合
        feature_dict: 后续各规则打分会用到的字段
        """
        C = []

        # 预取一些映射便于查询
        orders_df = state.orders_df
        time_rj_dict = state.time_rj_dict
        time_mrj_dict = state.time_mrj_dict
        task_r_dict = state.task_r_dict

        # 遍历所有 (j, m) 对 #action1对应工序类型-机器对的选择
        for action1, (j, m) in state.action_dict.items():
            if not state.eligible_fluid[j, m]:
                continue

            (r, j) = state.kind_task_tuple[j]
            machine = state.machine_tuple[m]

            # 该工序队列里的候选订单
            order_ids = state.kind_task_idle_id_list.get((r, j), [])
            if not order_ids:
                continue

            # 针对队列内每个订单，构造打分需要的特征
            for i, order_id in enumerate(order_ids):
                # 订单到达时间（FIFO 用）
                time_arrive = orders_df.loc[orders_df["order_id"] == order_id, "arrival_time"].values
                arrival_time = float(time_arrive[0]) if len(time_arrive) else 0.0

                # 该候选 (m,(r,j)) 的加工时间（SPT 用）
                time_process_mrj = float(time_mrj_dict.get(machine, {}).get((r, j), np.inf))

                # 当前工序在序列中的位置、剩余工序数（MOR 用）
                stages = list(task_r_dict[r])
                if j in stages:
                    idx_j = stages.index(j)
                else:
                    idx_j = 0
                rem_ops = len(stages) - idx_j  # 包含当前工序

                # 剩余工作量（MWKR 用）：从当前工序起按平均加工时间累加
                rem_work = 0.0
                for s in stages[idx_j:]:
                    rem_work += float(time_rj_dict.get((r, s), 0.0))

                feat = {
                    "arrival": arrival_time,   # FIFO
                    "p": time_process_mrj,     # SPT
                    "rem_ops": rem_ops,        # MOR
                    "rem_work": rem_work,      # MWKR
                }
                C.append((action1, i, (j, m, i), feat))

        return C

    def _score(self, feat):
        if self.rule == "fifo":
            return -feat["arrival"]
        if self.rule == "spt":
            return -feat["p"]
        if self.rule == "mor":
            return +feat["rem_ops"]
        if self.rule == "mwkr":
            return +feat["rem_work"]
        if self.rule == "rand":
            return random.random()
        raise RuntimeError("未实现的规则")

    def act(self, state):
        """
        返回 (action_index, log_prob, action1, action2)
        """
        C = self._collect_candidates(state)
        if not C:
            return None, None, None, None

        if self.rule == "rand":
            action1, action2, _, _ = random.choice(C)
        else:
            scores = [self._score(c[3]) for c in C]
            best = int(np.argmax(scores))
            action1, action2, _, _ = C[best]

        # 兼容 PPO 调用处的占位（不用于训练）
        action_index = torch.tensor(0, dtype=torch.long, device=self.device)
        log_prob = torch.tensor(0.0, device=self.device)
        return action_index, log_prob, action1, action2


def build_policy(name: str, device: str = "cpu") -> RulePolicy:
    return RulePolicy(rule=name, device=device)
