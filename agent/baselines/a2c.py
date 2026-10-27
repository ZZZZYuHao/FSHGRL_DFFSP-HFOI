"""Advantage actor-critic baseline (new code, ported from the former compare/A2C_model.py).

Same network as the PPO agent (HGNNScheduler, or MLPScheduler for model_type mlp) and the same rollout
collection; the update is the former one: Monte-Carlo discounted returns per trajectory, normalised first
to [0, 1] and then to zero mean / unit variance, one gradient step over the whole rollout with the loss
``-A_coeff * mean(log_prob * advantage) + vf_coeff * MSE(V, R) - entropy_coeff * mean(entropy)``. The step
is accumulated over shape-bucketed minibatches (identical gradient, bounded memory) and clipped by
``max_grad_norm``. No numbers of the original implementation exist to compare against.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from agent.baselines.common import monte_carlo_returns
from agent.ppo import _resolve_scheduler_class


class A2C:
    def __init__(self, model_paras: dict, train_paras: dict):
        self.lr = float(train_paras["lr"])
        self.current_lr = self.lr
        self.betas = tuple(train_paras.get("betas", (0.9, 0.999)))
        self.gamma = float(train_paras["gamma"])
        self.A_coeff = float(train_paras["A_coeff"])
        self.vf_coeff = float(train_paras["vf_coeff"])
        self.entropy_coeff = float(train_paras["entropy_coeff"])
        self.max_grad_norm = float(train_paras.get("max_grad_norm", 0.5))
        self.device = model_paras["device"]
        scheduler_class = _resolve_scheduler_class(train_paras)
        self.policy = scheduler_class(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

    def set_learning_rate(self, lr: float) -> None:
        self.current_lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr

    def update_rollout(self, buffer) -> dict:
        zero = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        count = len(buffer)
        if count == 0:
            return zero
        returns = torch.tensor(monte_carlo_returns(buffer.rewards, buffer.dones, buffer.trajectory_ids, self.gamma),
                               dtype=torch.float32, device=self.device)
        returns = (returns - returns.min()) / (returns.max() - returns.min() + 1e-5)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        self.policy.train()
        self.optimizer.zero_grad()
        sums = dict(zero)
        for indices in buffer.iter_minibatches(shuffle=False):
            obs_batch = [buffer.obs[index] for index in indices]
            actions = [buffer.env_actions[index] for index in indices]
            log_probs, state_values, entropies = self.policy.evaluate_batch(obs_batch, actions)
            batch_returns = returns[indices]
            advantages = (batch_returns - state_values).detach()
            policy_loss = -(log_probs * advantages).mean()
            value_loss = F.mse_loss(state_values, batch_returns)
            entropy = entropies.mean()
            loss = self.A_coeff * policy_loss + self.vf_coeff * value_loss - self.entropy_coeff * entropy
            weight = len(indices) / count
            (loss * weight).backward()
            sums["loss"] += loss.item() * weight
            sums["policy_loss"] += policy_loss.item() * weight
            sums["value_loss"] += value_loss.item() * weight
            sums["entropy"] += entropy.item() * weight
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        return sums
