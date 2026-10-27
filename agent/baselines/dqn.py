"""DQN / Double DQN baselines (new code, ported from the former compare/DQN_model.py and DDQN_model.py).

``QScheduler`` scores every candidate action (eligible machine-operation pair x queued order) of an
observation with a scalar Q value: the graph encoder of ``agent/baselines/encoders.py`` plus an MLP head
on the same per-candidate feature the PPO actor uses. ``DQNAgent`` holds the online network (``policy``),
the target network (``policy_old``, hard-copied every ``target_update_steps`` gradient steps or soft-updated
with ``target_tau``) and the TD update on replayed transitions (``td_loss`` mse as in the original file, or
huber). ``double=True`` evaluates the online argmax with the target network (Double DQN). The training loop
is ``agent/baselines/offpolicy.py``. No numbers of the original implementation exist to compare against.
"""
from __future__ import annotations

import copy
import random
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent.baselines.encoders import SparseGraphEncoder
from agent.mlp import MLPActor


class QScheduler(nn.Module):
    def __init__(self, model_paras: dict):
        super().__init__()
        self.device = model_paras["device"]
        self.encoder = SparseGraphEncoder(model_paras)
        self.q_head = MLPActor(model_paras["n_hidden_actor"], self.encoder.feature_dim,
                               model_paras["n_latent_actor"], 1).to(self.device)
        self.last_forward_stats = {}

    def set_device(self, device) -> None:
        self.device = torch.device(device)
        self.encoder.device = self.device
        self.to(self.device)

    def q_values(self, obs_batch) -> list:
        """``[(q [T], tokens), ...]``: one Q vector over the candidate tokens of each observation."""
        batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates = self.encoder.encode(obs_batch)
        out = []
        for batch_index, obs in enumerate(obs_batch):
            features, tokens = self.encoder.candidate_tokens(
                obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates,
                batch["current_rate"][batch_index])
            out.append((self.q_head(features.float()).squeeze(-1), tokens))
        return out

    def act_batch(self, obs_batch, sample: bool = False, epsilon: float = 0.0, rng: Optional[random.Random] = None):
        """Same return layout as the policy networks: ``(token indexes, log_probs, values, env_actions)``.

        ``sample=False``: argmax Q (greedy). ``epsilon > 0`` with ``rng``: epsilon-greedy with a uniform
        random candidate (training). ``sample=True``: Boltzmann sampling of softmax(Q) (eval.py --decode sample).
        """
        start_time = time.perf_counter()
        action_indexes, env_actions, token_counts = [], [], []
        for q, tokens in self.q_values(obs_batch):
            if rng is not None and epsilon > 0.0 and rng.random() < epsilon:
                index = rng.randrange(len(tokens))
            elif sample:
                index = int(Categorical(F.softmax(q, dim=0)).sample().item())
            else:
                index = int(torch.argmax(q).item())
            machine, task, rank = tokens[index]
            action_indexes.append(index)
            env_actions.append((machine * len(obs_batch[len(action_indexes) - 1]["raw_opes"]) + task, rank))
            token_counts.append(len(tokens))
        counts = torch.tensor(token_counts, dtype=torch.float32)
        self.last_forward_stats = {
            "batch_size": int(len(obs_batch)),
            "avg_action_tokens": float(counts.mean().item()) if counts.numel() else 0.0,
            "max_action_tokens": float(counts.max().item()) if counts.numel() else 0.0,
            "padding_ratio": 0.0,
            "forward_seconds": float(time.perf_counter() - start_time),
        }
        zeros = torch.zeros(len(obs_batch), dtype=torch.float32, device=self.device)
        return torch.tensor(action_indexes, dtype=torch.long, device=self.device), zeros, zeros, env_actions


class DQNAgent:
    def __init__(self, model_paras: dict, train_paras: dict, double: bool = False):
        self.lr = float(train_paras["lr"])
        self.current_lr = self.lr
        self.betas = tuple(train_paras.get("betas", (0.9, 0.999)))
        self.gamma = float(train_paras["gamma"])
        self.max_grad_norm = float(train_paras.get("max_grad_norm", 0.5))
        self.target_tau = float(train_paras.get("target_tau", 0.0) or 0.0)
        self.target_update_steps = max(int(train_paras.get("target_update_steps", 50)), 1)
        self.td_loss = str(train_paras.get("td_loss", "mse")).lower()
        if self.td_loss not in ("mse", "huber"):
            raise ValueError("td_loss must be 'mse' or 'huber'")
        self.double = bool(double)
        self.device = model_paras["device"]
        self.policy = QScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)           # target network
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.update_count = 0

    def set_learning_rate(self, lr: float) -> None:
        self.current_lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr

    def td_update(self, transitions: List) -> dict:
        """One gradient step on a minibatch of ``Transition`` records (offpolicy.py)."""
        self.policy.train()
        obs_batch = [t.obs for t in transitions]
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        q_list = self.policy.q_values(obs_batch)
        q_sa = torch.stack([q[t.token] for (q, _), t in zip(q_list, transitions)])
        targets = rewards.clone()
        next_positions = [i for i, t in enumerate(transitions) if not t.done and t.next_obs is not None]
        if next_positions:
            next_obs = [transitions[i].next_obs for i in next_positions]
            with torch.no_grad():
                target_q = self.policy_old.q_values(next_obs)
                if self.double:
                    online_q = self.policy.q_values(next_obs)
                    next_values = torch.stack([tq[int(torch.argmax(oq).item())]
                                               for (tq, _), (oq, _) in zip(target_q, online_q)])
                else:
                    next_values = torch.stack([tq.max() for tq, _ in target_q])
            targets[next_positions] = rewards[next_positions] + self.gamma * next_values
        loss = F.smooth_l1_loss(q_sa, targets) if self.td_loss == "huber" else F.mse_loss(q_sa, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.update_count += 1
        self._update_target()
        return {"loss": float(loss.item()), "td_error_abs": float((q_sa - targets).abs().mean().item()),
                "q_mean": float(q_sa.mean().item())}

    def _update_target(self) -> None:
        if self.target_tau > 0.0:
            with torch.no_grad():
                for target_param, param in zip(self.policy_old.parameters(), self.policy.parameters()):
                    target_param.data.mul_(1.0 - self.target_tau).add_(self.target_tau * param.data)
        elif self.update_count % self.target_update_steps == 0:
            self.policy_old.load_state_dict(self.policy.state_dict())

    def update(self, replay, rng: random.Random, steps: int, batch_size: int) -> dict:
        """``steps`` minibatch TD updates on uniformly sampled transitions; mean statistics for the log row."""
        zero = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                "clip_fraction": 0.0, "td_error_abs": 0.0, "q_mean": 0.0, "gradient_steps": 0}
        if len(replay) == 0 or steps <= 0:
            return zero
        totals = {"loss": 0.0, "td_error_abs": 0.0, "q_mean": 0.0}
        for _ in range(int(steps)):
            batch = replay.sample(rng, min(int(batch_size), len(replay)))
            stats = self.td_update(batch)
            for key in totals:
                totals[key] += stats[key]
        count = max(int(steps), 1)
        return {**zero, "loss": totals["loss"] / count, "value_loss": totals["loss"] / count,
                "td_error_abs": totals["td_error_abs"] / count, "q_mean": totals["q_mean"] / count,
                "gradient_steps": int(steps)}


class DDQNAgent(DQNAgent):
    """Double DQN: the online network chooses the next action, the target network evaluates it."""

    def __init__(self, model_paras: dict, train_paras: dict, double: bool = True):
        super().__init__(model_paras, train_paras, double=True)
