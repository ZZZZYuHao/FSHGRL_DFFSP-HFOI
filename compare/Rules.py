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


class RulePolicy:
    def __init__(self, rule: str, device: str = "cpu"):
        rule = rule.lower()
        assert rule in {"mor", "fifo", "mwkr", "spt"}, f"未知规则: {rule}"
        self.rule = rule
        self.device = device
        #self._arrival_map = None

    #def _ensure_arrival_map(self, orders_df):
    #    if self._arrival_map is None:
    #        self._arrival_map = dict(
    #            zip(orders_df["order_id"].values.tolist(),
    #                orders_df["arrival_time"].astype(float).values.tolist())
    #        )

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

        # self._ensure_arrival_map(orders_df)

        # 遍历所有 (j, m) 对 #action1对应工序类型-机器对的选择
        for action1, (j, m) in state.action_dict.items():
            if not state.eligible[j, m]:
                continue

            (r, j) = state.kind_task_tuple[j]
            # assert j_actual == j, "kind_task_tuple 与 action_dict 的 j 不一致 "

            machine = state.machine_tuple[m]
            # 该工序队列里的候选订单
            order_ids = state.kind_task_idle_id_list.get((r, j), [])
            if not order_ids:
                continue
            """
            # SPT
            time_process_mrj = float(time_mrj_dict.get(machine, {}).get((r, j), np.inf))
            stages = list(task_r_dict[j])
            idx_j = stages.index(j) if j in stages else 0
            rem_ops = len(stages) - idx_j

            #MWKR
            rem_work = 0.0
            for s in stages:
                rem_work += float(time_rj_dict.get((r, s), 0.0))

            for idx_in_queue, order_id in enumerate(order_ids):
                arrival_time = float(self._arrival_map.get(order_id, 0.0))
                feat = {
                    "arrival": arrival_time,   # FIFO
                    "p": time_process_mrj,     # SPT
                    "rem_ops": rem_ops,        # MOR
                    "rem_work": rem_work,      # MWKR
                    "queue_idx": idx_in_queue,
                }
                C.append((action1, idx_in_queue, (j, m, idx_in_queue), feat))
            """

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
                    "arrival": arrival_time,
                    "p": time_process_mrj,
                    "rem_ops": rem_ops,
                    "rem_work": rem_work,
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
        raise RuntimeError(f"未实现的规则")

    def act(self, state):
        """
        返回 (action_index, log_prob, action1, action2)
        """
        C = self._collect_candidates(state)
        if not C:
            return None, None, None, None
        """
        primary = np.array([self._primary_score[c[3]] for c in C])
        arrival = np.array([-c[3]["arrival"] for c in C], dtype=float)
        qpos = np.array([-c[3]["queue_idx"] for c in C], dtype=float)
        stacked = np.column_stack([primary, arrival, qpos])
        best = int(np.lexsort(stacked.T[::-1]))
        """
        scores = [self._score(c[3]) for c in C]
        best = int(np.argmin(scores))

        action1, action2, _, _ = C[best]

        # 兼容 PPO 调用处的占位（不用于训练）
        action_index = torch.tensor(0, dtype=torch.long, device=self.device)
        log_prob = torch.tensor(0.0, device=self.device)
        return action_index, log_prob, action1, action2

#def build_policy(name: str, device: str = "cpu") -> RulePolicy:
#    return RulePolicy(rule=name, device=device)


def _score_by(rule: str, feat: dict) -> float:
    if rule == "fifo":
        return -feat["arrival"]
    if rule == "spt":
        return -feat["p"]
    if rule == "mor":
        return +feat["rem_ops"]
    if rule == "mwkr":
        return +feat["rem_work"]
    raise RuntimeError(f"未实现的规则:{rule}")

class RandomRulePolicy:
    """
    随机规则策略：
    ‘per_episode’：整次算例随机选择1个规则
    ‘per_step’：每个决策点随机选择1个规则
    """
    def __init__(self, device: str = "cpu", mode: str = "per_episode", seed: int | None = None):
        assert mode in {"per_episode", "per_step"}, f"未知随机模式：{mode}"
        self.device = device
        self.mode = mode
        self.rules = ("mor", "fifo", "mwkr", "spt")
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        self.inner = None
        self.chosen_rule = None

        if self.mode == "per_episode":
            self.chosen_rule = self.rng.choice(self.rules)
            self.inner = RulePolicy(rule=self.chosen_rule, device=self.device)
            print(f'[RandomRulePolicy] 本算例随机选定的规则:{self.chosen_rule}')

    def act(self, state):
        """
        返回 (action_index, log_prob, action1, action2)
        """
        # per_episode
        if self.mode == "per_episode":
            return self.inner.act(state)

        # per_step
        C = RulePolicy("fifo", self.device)._collect_candidates(state)
        if not C:
            return None, None, None, None

        rule_now = self.rng.choice(self.rules)
        scores = [_score_by(rule_now, c[3]) for c in C]
        best = int(np.argmax(scores))
        action1, action2, _, _ = C[best]

        # 兼容 PPO 调用处的占位（不用于训练）
        action_index = torch.tensor(0, dtype=torch.long, device=self.device)
        log_prob = torch.tensor(0.0, device=self.device)
        return action_index, log_prob, action1, action2


def build_policy(name: str, device: str = "cpu"):
    name = name.lower()
    if name in {"mor", "fifo", "mwkr", "spt"}:
        return RulePolicy(rule=name, device=device)
    if name in {"rand", "random"}:
        return RandomRulePolicy(device=device, mode="per_episode")
    if name in {"rand_step", "random_step"}:
        return RandomRulePolicy(device=device, mode="per_step")
    raise AssertionError(f"未知规则:{name}")
