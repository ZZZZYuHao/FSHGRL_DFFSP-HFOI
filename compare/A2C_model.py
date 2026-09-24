"""
A2C代理类
"""
from agent.HGHH_model import HGNNScheduler
import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Any

class A2C:
    def __init__(self, model_paras, train_paras):
        self.lr = train_paras["lr"]
        self.gamma = train_paras["gamma"]
        self.A_coeff = train_paras["A_coeff"]
        self.vf_coeff = train_paras["vf_coeff"]
        self.entropy_coeff = train_paras["entropy_coeff"]
        self.device = model_paras["device"]

        self.policy = HGNNScheduler(model_paras).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # 计算折扣奖励
        discounted_rewards = []
        discounted_reward = 0.0
        for reward, terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if terminal:
                discounted_reward = 0.0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)
        rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
        rewards_tensor = (rewards_tensor - rewards_tensor.min()) / (rewards_tensor.max() - rewards_tensor.min() + 1e-5)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean())/(rewards_tensor.std()+1e-5)

        action_logprobs = []
        state_values = []
        dist_entropys = []
        for i in range(len(memory.states)):
            state = memory.states[i]
            action_logprob, state_value, dist_entropy = self.policy.evaluate(state,
                                                                             memory.ope_ma_adj[i],
                                                                             memory.ope_pre_adj[i],
                                                                             memory.ope_sub_adj[i],
                                                                             memory.raw_opes[i],
                                                                             memory.raw_mas[i],
                                                                             memory.proc_time[i],
                                                                             memory.eligible[i],
                                                                             memory.action_indexes[i])
            action_logprobs.append(action_logprob)
            state_values.append(state_value)
            dist_entropys.append(dist_entropy)

        action_logprobs_tensor = torch.stack(action_logprobs).squeeze()
        state_values_tensor = torch.stack(state_values)
        dist_entropys_tensor = torch.stack(dist_entropys).squeeze()

        advantages = (rewards_tensor - state_values_tensor).detach()

        loss = (- self.A_coeff * (action_logprobs_tensor * advantages).mean()
                + self.vf_coeff * self.MseLoss(state_values_tensor, rewards_tensor)
                - self.entropy_coeff * dist_entropys_tensor.mean())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
