"""Helpers shared by the learned baselines (new code)."""
from __future__ import annotations

import copy
from typing import List

import torch


def monte_carlo_returns(rewards: List[float], dones: List[bool], trajectory_ids: List[int], gamma: float) -> List[float]:
    """Discounted return of every transition, reset at terminal transitions, per trajectory (as the former
    compare/A2C_model.py did for a single environment)."""
    returns = [0.0 for _ in rewards]
    by_trajectory: dict = {}
    for index, trajectory_id in enumerate(trajectory_ids):
        by_trajectory.setdefault(trajectory_id, []).append(index)
    for indices in by_trajectory.values():
        running = 0.0
        for index in reversed(indices):
            if dones[index]:
                running = 0.0
            running = float(rewards[index]) + gamma * running
            returns[index] = running
    return returns


def prepare_policy(policy, rollout_device: torch.device, update_device: torch.device):
    """Rollout copy of a network on ``rollout_device`` in eval mode (the network itself when the devices match)."""
    if rollout_device == update_device:
        policy.eval()
        return policy
    clone = copy.deepcopy(policy)
    if hasattr(clone, "set_device"):
        clone.set_device(rollout_device)
    else:
        clone.to(rollout_device)
        clone.device = rollout_device
    clone.eval()
    return clone


LEGACY_ROW_KEYS = [
    "iteration", "instance_id", "instance_name", "instance_selection", "transitions", "episode_count",
    "episode_transition_mean", "episode_transition_std", "episode_seconds_mean", "episode_seconds_std", "reward",
    "mean_return", "fulfillment_rate_mean", "fulfillment_rate_std", "fulfillment_rate_min", "fulfillment_rate_max",
    "loss", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "learning_rate", "discard_count",
    "completion_count", "steps_per_second", "rollout_seconds", "update_seconds", "elapsed_seconds",
    "fluid_solve_count", "fluid_solve_seconds", "fluid_cache_hit_count", "fluid_fallback_count",
    "sparse_attention_enabled", "sparse_attention_scope", "policy_forward_count", "policy_forward_seconds",
    "avg_action_tokens", "max_action_tokens", "padding_ratio", "actor_batch_size_mean", "policy_act_seconds",
    "simulator_step_seconds", "obs_build_seconds", "rollout_device", "update_device", "cpu_memory_mb",
    "gpu_memory_mb", "gpu_temperature",
]


def legacy_metrics_row(iteration: int, instance_index: int, selected_instance, instance_selection: str, buffer,
                       update_metrics: dict, learning_rate: float, rollout_seconds: float, update_seconds: float,
                       elapsed: float, resources: dict, rollout_device, update_device, mean_return: float,
                       sparse_attention_enabled: bool, sparse_attention_scope: str) -> dict:
    """The 49-column train_metrics.csv row of the PPO trainer, for the baseline loops (same keys, same order)."""
    fulfillment_metrics = buffer.fulfillment_summary()
    episode_metrics = buffer.episode_summary()
    row = {
        "iteration": iteration,
        "instance_id": instance_index + 1,
        "instance_name": selected_instance.name,
        "instance_selection": instance_selection,
        "transitions": len(buffer),
        "episode_count": episode_metrics["episode_count"],
        "episode_transition_mean": episode_metrics["episode_transition_mean"],
        "episode_transition_std": episode_metrics["episode_transition_std"],
        "episode_seconds_mean": episode_metrics["episode_seconds_mean"],
        "episode_seconds_std": episode_metrics["episode_seconds_std"],
        "reward": float(sum(buffer.rewards)),
        "mean_return": float(mean_return),
        "fulfillment_rate_mean": fulfillment_metrics["fulfillment_rate_mean"],
        "fulfillment_rate_std": fulfillment_metrics["fulfillment_rate_std"],
        "fulfillment_rate_min": fulfillment_metrics["fulfillment_rate_min"],
        "fulfillment_rate_max": fulfillment_metrics["fulfillment_rate_max"],
        "loss": float(update_metrics.get("loss", 0.0)),
        "policy_loss": float(update_metrics.get("policy_loss", 0.0)),
        "value_loss": float(update_metrics.get("value_loss", 0.0)),
        "entropy": float(update_metrics.get("entropy", 0.0)),
        "approx_kl": float(update_metrics.get("approx_kl", 0.0)),
        "clip_fraction": float(update_metrics.get("clip_fraction", 0.0)),
        "learning_rate": learning_rate,
        "discard_count": buffer.discard_count,
        "completion_count": buffer.completion_count,
        "steps_per_second": len(buffer) / max(elapsed, 1e-8),
        "rollout_seconds": rollout_seconds,
        "update_seconds": update_seconds,
        "elapsed_seconds": elapsed,
        "fluid_solve_count": buffer.fluid_solve_count,
        "fluid_solve_seconds": buffer.fluid_solve_seconds,
        "fluid_cache_hit_count": buffer.fluid_cache_hit_count,
        "fluid_fallback_count": buffer.fluid_fallback_count,
        "sparse_attention_enabled": bool(sparse_attention_enabled),
        "sparse_attention_scope": str(sparse_attention_scope),
        "policy_forward_count": buffer.policy_forward_count,
        "policy_forward_seconds": buffer.policy_forward_seconds,
        "avg_action_tokens": buffer.avg_action_tokens,
        "max_action_tokens": buffer.max_action_tokens,
        "padding_ratio": buffer.padding_ratio,
        "actor_batch_size_mean": buffer.actor_batch_size_mean,
        "policy_act_seconds": buffer.policy_forward_seconds,
        "simulator_step_seconds": buffer.simulator_step_seconds,
        "obs_build_seconds": buffer.obs_build_seconds,
        "rollout_device": str(rollout_device),
        "update_device": str(update_device),
        "cpu_memory_mb": resources["cpu_memory_mb"],
        "gpu_memory_mb": resources["gpu_memory_mb"],
        "gpu_temperature": resources["gpu_temperature"],
    }
    assert list(row.keys()) == LEGACY_ROW_KEYS
    return row
