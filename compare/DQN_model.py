# DQN_model.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Tuple, List

from agent.hgnn import GATedge, MLPsim
from agent.HGHH_model import MLPs
from agent.mlp import MLPActor  # 复用你的 MLP 头，输出 1 维 Q(s,a)

class DQNScheduler(nn.Module):
    """
    与 HGNNScheduler 基本一致，但将 actor 输出作为 Q(s,a)（标量）。
    仍然构造 (task, machine, order) 的动作字典以保持与 env.step 的对齐。
    """
    def __init__(self, model_paras):
        super().__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]
        self.out_size_ma = model_paras["out_size_ma"]
        self.in_size_ope = model_paras["in_size_ope"]
        self.out_size_ope = model_paras["out_size_ope"]
        self.hidden_size_ope = model_paras["hidden_size_ope"]
        self.n_latent_actor = model_paras["n_latent_actor"]
        self.n_hidden_actor = model_paras["n_hidden_actor"]
        self.num_heads: List[int] = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]
        self.epsilon = model_paras.get("epsilon", 0.2)

        # GAT: machines
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma),
                                         self.out_size_ma, self.num_heads[0],
                                         feat_drop=self.dropout,
                                         attn_drop=self.dropout,
                                         activation=F.elu))
        for i in range(1, len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma),
                                             self.out_size_ma, self.num_heads[i],
                                             feat_drop=self.dropout,
                                             attn_drop=self.dropout,
                                             activation=F.elu))
        # OPE embeddings
        self.get_operations = nn.ModuleList()
        self.get_operations.append(
            MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                             self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout)
        )
        for i in range(len(self.num_heads) - 1):
            self.get_operations.append(
                MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                 self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout)
            )

        # Q-head：对每个候选动作的特征输出标量 Q
        # 输入维度与你 PPO 的 actor 相同：ope + ma + pooled(ope,ma) + due + rate
        # 但为避免配参错，这里直接在前向里按拼接结果“动态”喂给 MLPActor
        # 先放个占位 in_dim，实际不会被用（Torch 允许后续动态权重应用）
        self.q_head = MLPActor(self.n_hidden_actor,
                               input_dim=(self.out_size_ope + self.out_size_ma) * 2 + 2,
                               hidden_dim=self.n_latent_actor,
                               output_dim=1).to(self.device)

        self.current_action_dict = None

    # ===== 与 HGNNScheduler 相同的工具 =====
    def feature_normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - torch.mean(data)) / (torch.std(data) + 1e-5)

    def get_normalized(self,
                       raw_opes: torch.Tensor,
                       raw_mas: torch.Tensor,
                       proc_time: torch.Tensor):
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True)
        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
        proc_time_norm = self.feature_normalize(proc_time)
        norm_opes = (raw_opes - mean_opes) / (std_opes + 1e-5)
        norm_mas = (raw_mas - mean_mas) / (std_mas + 1e-5)
        return norm_opes, norm_mas, proc_time_norm

    # ===== 关键：计算 Q(s, ·) 与合法动作 mask =====
    @torch.no_grad()
    def build_order_due_tensor_(self, state, eligible, ope_ma_adj):
        max_long_due_date_list = max(state.kind_task_idle_id_due_date_list.values(), key=len) if len(state.kind_task_idle_id_due_date_list)>0 else []
        max_long = len(max_long_due_date_list) if len(max_long_due_date_list)>0 else 1
        eligibles = eligible.unsqueeze(-1).expand(-1, -1, -1, max_long).clone()
        h_ords_tensor = torch.full((len(state.kind_task_tuple), max_long),
                                   0, device=self.device, dtype=torch.float32)
        for key, values in state.kind_task_idle_id_due_date_list.items():
            kind_task_index = state.kind_task_tuple.index(key)
            for index, due_date in enumerate(values):
                h_ords_tensor[kind_task_index, index] = due_date
            if 0 < len(values) < max_long:
                eligibles[:, kind_task_index, :, len(values):] = False
        mean_ords_tensor = torch.mean(h_ords_tensor) if h_ords_tensor.numel()>0 else torch.tensor(0., device=self.device)
        std_ords_tensor = torch.std(h_ords_tensor) if h_ords_tensor.numel()>0 else torch.tensor(1., device=self.device)
        h_ords_tensor = (h_ords_tensor - mean_ords_tensor) / (std_ords_tensor + 1e-5)
        h_ords_padding = h_ords_tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-3).expand(-1, -1, ope_ma_adj.size(-1), -1, -1)
        return h_ords_padding, eligibles, max_long

    def _q_forward(self, state, require_grad: bool = False):
        """
        复用你 PPO 的 get_action_prob 的前半段，构造每个候选动作的特征张量 h_actions，
        然后用 q_head 计算 Q(s,a)。返回 q_values [1, A]，以及合法动作 mask [1, A]。
        """
        # --------- 原始特征 ----------
        raw_opes = state.feat_opes.unsqueeze(0).to(self.device)
        raw_mas  = state.feat_mas.unsqueeze(0).to(self.device)
        proc_time = state.proc_times.unsqueeze(0).to(self.device)

        # 归一化
        norm_opes, norm_mas, norm_proc = self.get_normalized(raw_opes, raw_mas, proc_time)

        # 邻接与合法
        ope_ma_adj = state.ope_ma_adj.unsqueeze(0).to(self.device)
        ope_pre_adj = state.ope_pre_adj.unsqueeze(0).to(self.device)
        ope_sub_adj = state.ope_sub_adj.unsqueeze(0).to(self.device)
        eligible = state.eligible_fluid.unsqueeze(0).to(self.device)  # 仍沿用你的流体掩码

        # HGNN L 层传播
        features = (norm_opes, norm_mas, norm_proc)
        L = len(self.num_heads)
        for i in range(L):
            h_mas = self.get_machines[i](ope_ma_adj, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, features)
            features = (h_opes, features[1], features[2])

        # pooling
        h_mas_pooled = torch.mean(h_mas, dim=-2)   # [1, Dm]
        h_opes_pooled = torch.mean(h_opes, dim=-2) # [1, Do]

        # 订单 due 通道与动作空间展开
        h_ords_padding, eligibles, max_long = self.build_order_due_tensor_(state, eligible, ope_ma_adj)

        # 扩展维度以构造动作特征 (O, M, L)
        h_opes_padding = h_opes.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, ope_ma_adj.size(-1), max_long, -1)
        h_mas_padding  = h_mas.unsqueeze(-3).unsqueeze(-2).expand(-1, ope_ma_adj.size(-2), -1, max_long, -1)

        B = 1
        O = ope_ma_adj.size(-2)
        M = ope_ma_adj.size(-1)
        L_ord = max_long

        h_mas_pooled_padding = h_mas_pooled.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, O, M, L_ord, -1)
        h_opes_pooled_padding= h_opes_pooled.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, O, M, L_ord, -1)
        h_rate_padding = torch.tensor(state.current_order_completed_rate, device=self.device).view(1,1,1,1,1).expand_as(h_ords_padding)

        # 拼接动作特征：与你 PPO 一致
        h_actions = torch.cat((h_opes_padding, h_mas_padding,
                               h_opes_pooled_padding, h_mas_pooled_padding,
                               h_ords_padding, h_rate_padding), dim=-1).transpose(1,2)  # [1, M, O, L, D]

        # 建立动作索引映射（与 PPO 一致）：base 为 (task_idx, machine_idx) 的索引，再 × 订单维
        self.current_action_dict = {
            action_index: (action1, action2)
            for action_index, (action1, action2) in enumerate(
                [(i, j) for i in state.action_dict.keys() for j in range(max_long)]
            )
        }
        # 合法掩码
        mask = eligibles.transpose(1, 2).flatten(1)  # [1, M*O*L]

        # 计算 Q(s,a)
        # 注意：MLPActor 支持高维输入，只要最后一维对齐
        q_vals = self.q_head(h_actions.float()).flatten(1)  # [1, M*O*L]
        # 对非法动作给一个极小值，避免被 argmax 选中
        q_vals[~mask] = float('-1e9')
        return q_vals, mask, max_long

    def act(self, state: object, memory, epoch, flag_sample: bool=True):
        """
        ε-greedy：用 policy_old（目标网）选择动作，返回主程序所需四元组。
        """
        with torch.no_grad():
            q_vals, mask, max_long = self._q_forward(state, require_grad=False)

        # epsilon 线性衰减（与 PPO 那套一致）
        epsilon = max(self.epsilon * (500 - epoch) / 500, 0.01)
        if random.random() < epsilon:
            valid_idx = torch.nonzero(mask[0], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                action_index = torch.tensor(0, device=self.device)
            else:
                ridx = torch.randint(0, valid_idx.numel(), (1,)).item()
                action_index = valid_idx[ridx]
            log_prob = torch.tensor(0.0, device=self.device)
        else:
            action_index = torch.argmax(q_vals, dim=1)[0]
            log_prob = torch.tensor(0.0, device=self.device)

        # 映射到 (action1, action2)
        a = int(action_index.item())
        action1, action2 = self.current_action_dict[a]
        return action_index, log_prob, action1, action2


class DQN:
    """
    对齐 PPO_model.PPO 的接口：
    - 有 policy（在线网）与 policy_old（目标网）
    - 有 update(memory) -> loss
    - 采样期主程序依然调用 policy_old.act(...)
    """
    def __init__(self, model_paras, train_paras):
        self.lr = train_paras["lr"]
        self.betas = train_paras["betas"]
        self.gamma = train_paras["gamma"]
        self.device = model_paras["device"]
        self.target_update_tau = train_paras.get("target_tau", None)  # 若想软更新，可给 0<tau<=1
        self.hard_update_every = train_paras.get("target_update_steps", 1)  # 多少次 update 做一次硬更新

        self.step_counter = 0

        self.policy = DQNScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy).to(self.device)  # 作为目标网络
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MSE = nn.MSELoss()

    @torch.no_grad()
    def _max_q_next(self, next_state):
        q_next, mask_next, _ = self.policy_old._q_forward(next_state, require_grad=False)
        # 如果全部非法，就返回 0（避免 -inf）
        if (~torch.isfinite(q_next)).all() or (mask_next.sum() == 0):
            return torch.tensor(0.0, device=self.device)
        return torch.max(q_next, dim=1)[0][0]

    def update(self, memory) -> float:
        """
        用 memory.states / action_indexes / rewards / is_terminals
        重建 (s, a, r, s')，做一次全量 TD 更新。
        返回总 loss（标量）。
        """
        N = len(memory.states)
        if N == 0:
            return 0.0

        total_loss = 0.0
        self.optimizer.zero_grad()

        for i in range(N):
            s = memory.states[i]
            a = memory.action_indexes[i].to(self.device) if i < len(memory.action_indexes) else torch.tensor(0, device=self.device)
            r = torch.as_tensor(memory.rewards[i] if i < len(memory.rewards) else 0.0, dtype=torch.float32, device=self.device)
            done = bool(memory.is_terminals[i]) if i < len(memory.is_terminals) else True

            # Q(s,a)
            q_vals, _, _ = self.policy._q_forward(s, require_grad=True)
            q_sa = q_vals[0, a.item()]

            # 目标 y = r + gamma * max_a' Q_target(s', a')
            if i < N - 1 and not done:
                next_state = memory.states[i + 1]
                max_q_next = self._max_q_next(next_state)
                y = r + self.gamma * max_q_next
            else:
                y = r

            loss = self.MSE(q_sa, y)
            loss.backward()
            total_loss += float(loss.item())

        self.optimizer.step()

        # 目标网络更新
        self.step_counter += 1
        if self.target_update_tau is not None:
            # 软更新
            with torch.no_grad():
                for p, q in zip(self.policy.parameters(), self.policy_old.parameters()):
                    q.data.mul_(1.0 - self.target_update_tau).add_(self.target_update_tau * p.data)
        else:
            # 硬更新
            if self.step_counter % self.hard_update_every == 0:
                self.policy_old.load_state_dict(self.policy.state_dict())

        return total_loss
