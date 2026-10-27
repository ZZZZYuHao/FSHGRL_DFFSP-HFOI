"""Learned baselines (A2C, DQN, DDQN): construction, acting on real observations, one update, checkpoints."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.baselines.common import monte_carlo_returns  # noqa: E402
from agent.baselines.offpolicy import ReplayBuffer, collect_offpolicy, epsilon_for  # noqa: E402
from agent.checkpoint import load_checkpoint, load_policy, save_checkpoint  # noqa: E402
from agent.factory import build_agent, build_policy_network  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402
from environment.simulator import TrainingSimulator  # noqa: E402
from rollout import collect_vectorized_rollouts  # noqa: E402
from train import build_model_paras, load_config  # noqa: E402

INSTANCE = ROOT / "data" / "instances" / "benchmark" / "DDT500_num50" / "DDT500_num50.txt"


def _config(algorithm: str):
    config = load_config(None)
    train_paras = dict(config["train_paras"])
    train_paras.update({"algorithm": algorithm, "fluid_mode": "off", "num_envs": 1, "num_workers": 1,
                        "minibatch_size": 64, "viz": False})
    return config, train_paras


def test_monte_carlo_returns_reset_at_terminals():
    returns = monte_carlo_returns([1.0, 1.0, 1.0, 2.0], [False, True, False, True], [0, 0, 1, 1], 0.5)
    assert returns == [1.5, 1.0, 2.0, 2.0]


def test_epsilon_decay_matches_the_original_formula():
    paras = {"epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay_iterations": 500}
    assert epsilon_for(0, paras) == 1.0 and abs(epsilon_for(250, paras) - 0.5) < 1e-12
    assert epsilon_for(499, paras) == max(1.0 / 500, 0.01) and epsilon_for(10_000, paras) == 0.01


def test_a2c_update_and_sync():
    config, train_paras = _config("a2c")
    torch.manual_seed(0)
    agent = build_agent(build_model_paras(config, torch.device("cpu"), train_paras), train_paras)
    rollout = collect_vectorized_rollouts(agent.policy_old, INSTANCE, train_paras)
    rollout.minibatch_size = 64
    rollout.compute_returns_and_advantages(gamma=1.0, gae_lambda=0.95)
    before = {k: v.clone() for k, v in agent.policy.state_dict().items()}
    metrics = agent.update_rollout(rollout)
    assert all(torch.isfinite(torch.tensor(float(metrics[k]))) for k in ("loss", "policy_loss", "value_loss", "entropy"))
    assert any(not torch.equal(before[k], v) for k, v in agent.policy.state_dict().items())
    for key, value in agent.policy.state_dict().items():
        assert torch.equal(value, agent.policy_old.state_dict()[key])


def test_q_scheduler_actions_are_feasible_and_td_update_runs(tmp_path):
    for algorithm in ("dqn", "ddqn"):
        config, train_paras = _config(algorithm)
        torch.manual_seed(1)
        agent = build_agent(build_model_paras(config, torch.device("cpu"), train_paras), train_paras)
        assert agent.double == (algorithm == "ddqn")
        spec = parse_instance_file(INSTANCE)
        simulator = TrainingSimulator(spec, train_paras)
        obs = simulator.to_policy_obs()
        with torch.no_grad():
            tokens, log_probs, values, env_actions = agent.policy.act_batch([obs], sample=False)
        machine, task = env_actions[0][0] // spec.task_count, env_actions[0][0] % spec.task_count
        valid = {(int(m), int(t)) for m, t in obs["valid_pairs"].tolist()}
        assert (machine, task) in valid and 0 <= env_actions[0][1] < int(obs["order_counts"][task])
        assert tokens.shape == (1,) and log_probs.shape == (1,) and values.shape == (1,)

        replay = ReplayBuffer(capacity=1000)
        rng = random.Random(0)
        buffer = collect_offpolicy(agent.policy, spec, train_paras, 1, 0.5, rng, replay, {}, 0)
        assert len(replay) == len(buffer) == buffer.episode_transition_counts[0] > 0
        assert buffer.dones[-1] is True and replay._items[-1].done and replay._items[-1].next_obs is None
        assert replay._items[0].obs["proc_time"] is replay._items[1].obs["proc_time"]  # shared static tensor
        metrics = agent.update(replay, rng, steps=2, batch_size=16)
        assert metrics["gradient_steps"] == 2 and torch.isfinite(torch.tensor(metrics["loss"]))

        path = tmp_path / f"{algorithm}.pt"
        save_checkpoint(path, agent, 3, 0.5, train_paras, config)
        fresh = build_agent(build_model_paras(config, torch.device("cpu"), train_paras), train_paras)
        iteration, best = load_checkpoint(path, fresh, torch.device("cpu"))
        assert (iteration, best) == (4, 0.5)
        for key, value in agent.policy.state_dict().items():
            assert torch.equal(value, fresh.policy.state_dict()[key])
        policy, _, loaded_paras, bundle = load_policy(path, "cpu")
        assert type(policy) is type(agent.policy) and loaded_paras["algorithm"] == algorithm and bundle["iteration"] == 3


def test_factory_networks_match_algorithms():
    config, train_paras = _config("ppo")
    model_paras = build_model_paras(config, torch.device("cpu"), train_paras)
    from agent.baselines.dqn import QScheduler
    from agent.networks import HGNNScheduler

    assert isinstance(build_policy_network(model_paras, dict(train_paras, algorithm="ppo")), HGNNScheduler)
    assert isinstance(build_policy_network(model_paras, dict(train_paras, algorithm="a2c")), HGNNScheduler)
    assert isinstance(build_policy_network(model_paras, dict(train_paras, algorithm="dqn")), QScheduler)
