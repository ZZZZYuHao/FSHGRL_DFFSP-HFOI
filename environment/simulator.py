"""Discrete-event training simulator (moved verbatim from agent/training_simulator.py)."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from data.instance_io import InstanceSpec, parse_instance_file  # noqa: F401  (re-exported for compatibility)


class TrainingSimulator:
    STATUS_NOT_ARRIVED = 0
    STATUS_WAITING = 1
    STATUS_PROCESSING = 2
    STATUS_COMPLETED = 3
    STATUS_DISCARDED = 4
    _GUROBI_IMPORT_ATTEMPTED = False
    _GUROBI_MODULE = None
    _GUROBI_IMPORT_ERROR = None

    def __init__(self, spec: InstanceSpec, train_paras: dict):
        self.spec = spec
        self.order_top_k = int(train_paras.get("order_top_k", 5))
        self.discard_penalty = float(train_paras.get("discard_penalty", 1.0))
        self.progress_reward_coeff = float(train_paras.get("progress_reward_coeff", 0.0))
        self.fluid_mode = str(train_paras.get("fluid_mode", "heuristic")).lower()
        self.fluid_time_limit = float(train_paras.get("fluid_time_limit", 20))
        self.fluid_threads = int(train_paras.get("fluid_threads", 0))
        self.reset()

    def reset(self):
        self.current_time = float(self.spec.arrival_times.min()) if self.spec.total_orders else 0.0
        self.order_status = np.full(self.spec.total_orders, self.STATUS_NOT_ARRIVED, dtype=np.int8)
        self.order_stage = np.zeros(self.spec.total_orders, dtype=np.int16)
        self.machine_end_times = np.zeros(self.spec.machine_count, dtype=np.float32)
        self.machine_order = np.full(self.spec.machine_count, -1, dtype=np.int32)
        self.machine_stage = np.full(self.spec.machine_count, -1, dtype=np.int16)
        self.completed_count = 0
        self.discarded_count = 0
        self.transition_count = 0
        self.schedule_count = 0
        self.obs_build_seconds = 0.0
        self.step_seconds = 0.0
        self.fluid_solve_count = 0
        self.fluid_solve_seconds = 0.0
        self.fluid_cache_hit_count = 0
        self.fluid_fallback_count = 0
        self._fluid_cache = {}
        self._last_fluid_key = None
        self._fluid_recompute_required = False
        self._last_fluid_features = (
            np.zeros(self.spec.task_count, dtype=np.float32),
            np.zeros(self.spec.task_count, dtype=np.float32),
        )
        self._last_completion_rate = 0.0
        self._activate_arrivals()
        self._discard_expired_waiting_orders()
        self._advance_until_decision_or_done()
        return self

    @property
    def done(self) -> bool:
        return self.completed_count + self.discarded_count >= self.spec.total_orders

    @property
    def fulfillment_rate(self) -> float:
        return self.completed_count / max(self.spec.total_orders, 1)

    def _task_index(self, product: int, stage: int) -> int:
        return int(product) * self.spec.stage_count + int(stage)

    def _activate_arrivals(self):
        newly_arrived = (
            (self.order_status == self.STATUS_NOT_ARRIVED)
            & (self.spec.arrival_times <= self.current_time)
        )
        if newly_arrived.any():
            self.order_status[newly_arrived] = self.STATUS_WAITING
            self._fluid_recompute_required = True

    def _release_finished_machines(self):
        busy = self.machine_order >= 0
        finished_machines = np.where(busy & (self.machine_end_times <= self.current_time))[0]
        for machine in finished_machines:
            order_index = int(self.machine_order[machine])
            stage = int(self.machine_stage[machine])
            if self.order_status[order_index] == self.STATUS_PROCESSING:
                next_stage = stage + 1
                if next_stage >= self.spec.stage_count:
                    self.order_status[order_index] = self.STATUS_COMPLETED
                    self.completed_count += 1
                else:
                    self.order_stage[order_index] = next_stage
                    self.order_status[order_index] = self.STATUS_WAITING
            self.machine_order[machine] = -1
            self.machine_stage[machine] = -1
            self.machine_end_times[machine] = self.current_time

    def _discard_order(self, order_index: int):
        if self.order_status[order_index] in {
            self.STATUS_COMPLETED,
            self.STATUS_DISCARDED,
        }:
            return
        self.order_status[order_index] = self.STATUS_DISCARDED
        self.discarded_count += 1
        self._fluid_recompute_required = True
        for machine in np.where(self.machine_order == order_index)[0]:
            self.machine_order[machine] = -1
            self.machine_stage[machine] = -1
            self.machine_end_times[machine] = self.current_time

    def _discard_expired_waiting_orders(self):
        waiting = np.where(self.order_status == self.STATUS_WAITING)[0]
        for order_index in waiting:
            if self.current_time >= float(self.spec.due_dates[order_index]):
                self._discard_order(int(order_index))

    def _next_event_time(self) -> float | None:
        future_arrivals = self.spec.arrival_times[
            (self.order_status == self.STATUS_NOT_ARRIVED)
            & (self.spec.arrival_times > self.current_time)
        ]
        busy_ends = self.machine_end_times[
            (self.machine_order >= 0) & (self.machine_end_times > self.current_time)
        ]
        candidates = []
        if future_arrivals.size:
            candidates.append(float(future_arrivals.min()))
        if busy_ends.size:
            candidates.append(float(busy_ends.min()))
        return min(candidates) if candidates else None

    def _waiting_orders_by_task(self) -> Dict[int, List[int]]:
        grouped = {task_index: [] for task_index in range(self.spec.task_count)}
        waiting = np.where(self.order_status == self.STATUS_WAITING)[0]
        for order_index in waiting:
            product = int(self.spec.product_types[order_index])
            stage = int(self.order_stage[order_index])
            task_index = self._task_index(product, stage)
            grouped[task_index].append(int(order_index))
        for task_index, order_indices in grouped.items():
            order_indices.sort(key=lambda idx: float(self.spec.due_dates[idx]))
            if self.order_top_k > 0:
                grouped[task_index] = order_indices[:self.order_top_k]
        return grouped

    def _eligible_pairs(self, grouped_orders: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        pairs = []
        idle_machines = np.where(self.machine_order < 0)[0]
        for task_index, order_indices in grouped_orders.items():
            if not order_indices:
                continue
            _, stage = self.spec.task_to_product_stage[task_index]
            for machine in self.spec.stage_machines[stage]:
                if machine in idle_machines:
                    pairs.append((int(machine), int(task_index)))
        return pairs

    def has_decision(self) -> bool:
        grouped = self._waiting_orders_by_task()
        return bool(self._eligible_pairs(grouped))

    def _advance_until_decision_or_done(self):
        while not self.done and not self.has_decision():
            next_time = self._next_event_time()
            if next_time is None:
                waiting_or_processing = np.where(
                    (self.order_status == self.STATUS_WAITING)
                    | (self.order_status == self.STATUS_PROCESSING)
                    | (self.order_status == self.STATUS_NOT_ARRIVED)
                )[0]
                for order_index in waiting_or_processing:
                    self._discard_order(int(order_index))
                break
            self.current_time = next_time
            self._activate_arrivals()
            self._release_finished_machines()
            self._discard_expired_waiting_orders()

    @classmethod
    def _load_gurobi(cls):
        if not cls._GUROBI_IMPORT_ATTEMPTED:
            cls._GUROBI_IMPORT_ATTEMPTED = True
            try:
                import gurobipy as gp

                cls._GUROBI_MODULE = gp
            except Exception as exc:
                cls._GUROBI_IMPORT_ERROR = exc
                cls._GUROBI_MODULE = None
        return cls._GUROBI_MODULE

    def _heuristic_fluid_features(self, grouped_orders: Dict[int, List[int]]):
        allocations = np.zeros(self.spec.task_count, dtype=np.float32)
        rates = np.zeros(self.spec.task_count, dtype=np.float32)
        if self.fluid_mode == "off":
            return allocations, rates
        for task_index, (_, stage) in enumerate(self.spec.task_to_product_stage):
            order_count = len(grouped_orders.get(task_index, []))
            stage_machines = self.spec.stage_machines[stage]
            allocations[task_index] = float(len(stage_machines))
            if order_count == 0:
                continue
            rates[task_index] = float(sum(
                float(self.spec.proc_rates[task_index, machine].item())
                for machine in stage_machines
            ) / max(order_count, 1))
        return allocations, rates

    def _fluid_cache_key(self, grouped_orders: Dict[int, List[int]],
                         valid_pairs: List[Tuple[int, int]]):
        waiting_signature = tuple(
            (int(task_index), tuple(int(index) for index in order_indices))
            for task_index, order_indices in sorted(grouped_orders.items())
            if order_indices
        )
        idle_machines = tuple(int(index) for index in np.where(self.machine_order < 0)[0])
        unfinished_count = int(np.sum(
            (self.order_status != self.STATUS_COMPLETED)
            & (self.order_status != self.STATUS_DISCARDED)
        ))
        processing_signature = tuple(
            (int(machine), int(self.machine_stage[machine]))
            for machine in np.where(self.machine_order >= 0)[0]
        )
        return (
            idle_machines,
            waiting_signature,
            tuple(sorted((int(machine), int(task)) for machine, task in valid_pairs)),
            processing_signature,
            unfinished_count,
        )

    def _fallback_fluid_features(self, grouped_orders: Dict[int, List[int]]):
        self.fluid_fallback_count += 1
        return self._heuristic_fluid_features(grouped_orders)

    def _solve_cached_lp(self, grouped_orders: Dict[int, List[int]],
                         valid_pairs: List[Tuple[int, int]]):
        active_pairs = [
            (int(machine), int(task_index))
            for machine, task_index in valid_pairs
            if grouped_orders.get(int(task_index))
        ]
        if not active_pairs:
            return self._heuristic_fluid_features(grouped_orders)

        gp = self._load_gurobi()
        if gp is None:
            return self._fallback_fluid_features(grouped_orders)

        started = time.perf_counter()
        solve_seconds_recorded = False
        try:
            model = gp.Model("cached_fluid_lp")
            model.Params.OutputFlag = 0
            if self.fluid_time_limit > 0:
                model.Params.TimeLimit = self.fluid_time_limit
            if self.fluid_threads > 0:
                model.Params.Threads = self.fluid_threads

            x_vars = {}
            for machine, task_index in active_pairs:
                x_vars[(machine, task_index)] = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    name=f"x_{machine}_{task_index}",
                )
            min_rate = model.addVar(lb=0.0, name="min_rate")
            model.update()

            for machine in sorted({machine for machine, _ in active_pairs}):
                model.addConstr(
                    gp.quicksum(
                        variable
                        for (candidate_machine, _), variable in x_vars.items()
                        if candidate_machine == machine
                    ) <= 1.0,
                    name=f"machine_capacity_{machine}",
                )

            active_tasks = sorted({task_index for _, task_index in active_pairs})
            for task_index in active_tasks:
                demand = float(max(len(grouped_orders.get(task_index, [])), 1))
                service = gp.quicksum(
                    float(self.spec.proc_rates[task_index, machine].item()) * variable
                    for (machine, candidate_task), variable in x_vars.items()
                    if candidate_task == task_index
                )
                model.addConstr(
                    service >= min_rate * demand,
                    name=f"task_rate_floor_{task_index}",
                )

            model.setObjective(min_rate, gp.GRB.MAXIMIZE)
            self.fluid_solve_count += 1
            model.optimize()
            self.fluid_solve_seconds += time.perf_counter() - started
            solve_seconds_recorded = True

            if int(getattr(model, "SolCount", 0)) <= 0:
                return self._fallback_fluid_features(grouped_orders)

            allocations = np.zeros(self.spec.task_count, dtype=np.float32)
            rates = np.zeros(self.spec.task_count, dtype=np.float32)
            for (machine, task_index), variable in x_vars.items():
                value = max(float(variable.X), 0.0)
                allocations[task_index] += value
                rates[task_index] += float(self.spec.proc_rates[task_index, machine].item()) * value
            for task_index in active_tasks:
                rates[task_index] /= float(max(len(grouped_orders.get(task_index, [])), 1))
            return allocations, rates
        except Exception:
            if not solve_seconds_recorded:
                self.fluid_solve_seconds += time.perf_counter() - started
            return self._fallback_fluid_features(grouped_orders)

    def _fluid_features(self, grouped_orders: Dict[int, List[int]],
                        valid_pairs: List[Tuple[int, int]]):
        if self.fluid_mode == "off":
            return self._heuristic_fluid_features(grouped_orders)
        if self.fluid_mode != "cached_lp":
            return self._heuristic_fluid_features(grouped_orders)

        if not self._fluid_recompute_required:
            return self._last_fluid_features

        key = self._fluid_cache_key(grouped_orders, valid_pairs)
        if key in self._fluid_cache:
            self.fluid_cache_hit_count += 1
            features = self._fluid_cache[key]
        else:
            features = self._solve_cached_lp(grouped_orders, valid_pairs)
            self._fluid_cache[key] = features
        self._last_fluid_key = key
        self._last_fluid_features = features
        self._fluid_recompute_required = False
        return features

    def to_policy_obs(self) -> dict:
        start_time = time.perf_counter()
        grouped_orders = self._waiting_orders_by_task()
        valid_pairs = self._eligible_pairs(grouped_orders)
        task_count = self.spec.task_count
        machine_count = self.spec.machine_count
        max_orders = 1
        if grouped_orders:
            max_orders = max([len(value) for value in grouped_orders.values()] + [1])

        due_dates = torch.zeros((task_count, max_orders), dtype=torch.float32)
        order_mask = torch.zeros((task_count, max_orders), dtype=torch.bool)
        order_counts = torch.zeros((task_count,), dtype=torch.long)
        candidate_order_ids: List[List[int]] = [[] for _ in range(task_count)]
        for task_index, order_indices in grouped_orders.items():
            candidate_order_ids[task_index] = order_indices
            order_counts[task_index] = len(order_indices)
            for rank, order_index in enumerate(order_indices[:max_orders]):
                due_dates[task_index, rank] = float(self.spec.due_dates[order_index] - self.current_time)
                order_mask[task_index, rank] = True

        eligible = torch.zeros((task_count, machine_count), dtype=torch.bool)
        for machine, task_index in valid_pairs:
            eligible[task_index, machine] = True

        fluid_allocations, fluid_rates = self._fluid_features(grouped_orders, valid_pairs)
        raw_opes = torch.zeros((task_count, 12), dtype=torch.float32)
        for task_index, (product, stage) in enumerate(self.spec.task_to_product_stage):
            order_indices = grouped_orders.get(task_index, [])
            due_remaining = [
                float(self.spec.due_dates[index] - self.current_time)
                for index in order_indices
            ]
            stage_machines = self.spec.stage_machines[stage]
            raw_opes[task_index, 0] = len(stage_machines)
            raw_opes[task_index, 1] = int(eligible[task_index].sum().item())
            if due_remaining:
                raw_opes[task_index, 2] = float(np.mean(due_remaining))
                raw_opes[task_index, 3] = float(np.min(due_remaining))
                raw_opes[task_index, 4] = float(np.max(due_remaining))
                raw_opes[task_index, 5] = float(np.std(due_remaining))
            raw_opes[task_index, 6] = int(np.sum(
                (self.order_status != self.STATUS_COMPLETED)
                & (self.order_status != self.STATUS_DISCARDED)
                & (self.spec.product_types == product)
                & (self.order_stage <= stage)
            ))
            raw_opes[task_index, 7] = len(order_indices)
            raw_opes[task_index, 8] = float(fluid_allocations[task_index])
            raw_opes[task_index, 9] = float(fluid_rates[task_index])
            raw_opes[task_index, 10] = product
            raw_opes[task_index, 11] = stage

        raw_mas = torch.zeros((machine_count, 5), dtype=torch.float32)
        for machine in range(machine_count):
            idle = self.machine_order[machine] < 0
            candidate_tasks = [
                task_index
                for task_index, (_, stage) in enumerate(self.spec.task_to_product_stage)
                if machine in self.spec.stage_machines[stage]
            ]
            raw_mas[machine, 0] = len(candidate_tasks)
            raw_mas[machine, 1] = self.current_time if idle else float(self.machine_end_times[machine])
            raw_mas[machine, 2] = 1.0 if idle else 0.0
            raw_mas[machine, 3] = int(sum(eligible[task_index, machine].item() for task_index in candidate_tasks))
            raw_mas[machine, 4] = 0.0 if idle else 1.0

        valid_action_tokens = int(sum(
            len(grouped_orders.get(task_index, []))
            for _, task_index in valid_pairs
        ))
        padded_action_tokens = max(machine_count * task_count * max_orders, 1)
        self.obs_build_seconds += time.perf_counter() - start_time
        return {
            "raw_opes": raw_opes,
            "raw_mas": raw_mas,
            "proc_time": self.spec.proc_rates.clone(),
            "ope_ma_adj": self.spec.ope_ma_adj.clone(),
            "ope_pre_adj": self.spec.ope_pre_adj.clone(),
            "ope_sub_adj": self.spec.ope_sub_adj.clone(),
            "eligible": eligible,
            "due_dates": due_dates,
            "order_mask": order_mask,
            "order_counts": order_counts,
            "candidate_order_ids": candidate_order_ids,
            "valid_pairs": torch.tensor(valid_pairs, dtype=torch.long) if valid_pairs else torch.zeros((0, 2), dtype=torch.long),
            "current_order_completed_rate": self.fulfillment_rate,
            "num_tasks": task_count,
            "num_machines": machine_count,
            "max_orders": max_orders,
            "valid_pair_count": len(valid_pairs),
            "valid_order_count": int(order_mask.sum().item()),
            "valid_action_tokens": valid_action_tokens,
            "padded_action_tokens": padded_action_tokens,
            "order_top_k": self.order_top_k,
        }

    def step(self, env_action: Tuple[int, int]) -> float:
        start_time = time.perf_counter()
        if self.done:
            return 0.0

        previous_completed = self.completed_count
        previous_discarded = self.discarded_count
        previous_rate = self.fulfillment_rate
        grouped_orders = self._waiting_orders_by_task()
        action1, order_rank = int(env_action[0]), int(env_action[1])
        task_count = self.spec.task_count
        machine = action1 // task_count
        task_index = action1 % task_count
        valid_pairs = set(self._eligible_pairs(grouped_orders))
        if (machine, task_index) not in valid_pairs:
            if not valid_pairs:
                self._advance_until_decision_or_done()
                self.step_seconds += time.perf_counter() - start_time
                return 0.0
            machine, task_index = next(iter(valid_pairs))

        order_indices = grouped_orders.get(task_index, [])
        if not order_indices:
            self._advance_until_decision_or_done()
            self.step_seconds += time.perf_counter() - start_time
            return 0.0
        order_index = order_indices[min(max(order_rank, 0), len(order_indices) - 1)]
        product, stage = self.spec.task_to_product_stage[task_index]
        process_time = float(self.spec.process_times[product, machine])
        finish_time = self.current_time + process_time

        if finish_time > float(self.spec.due_dates[order_index]):
            self._discard_order(order_index)
        else:
            self.order_status[order_index] = self.STATUS_PROCESSING
            self.machine_order[machine] = order_index
            self.machine_stage[machine] = stage
            self.machine_end_times[machine] = finish_time
            self.schedule_count += 1

        self.transition_count += 1
        self._activate_arrivals()
        self._release_finished_machines()
        self._discard_expired_waiting_orders()
        self._advance_until_decision_or_done()
        total_orders = max(self.spec.total_orders, 1)
        reward = (
            (self.completed_count - previous_completed) / total_orders
            - self.discard_penalty * (self.discarded_count - previous_discarded) / total_orders
            + self.progress_reward_coeff * (self.fulfillment_rate - previous_rate)
        )
        self._last_completion_rate = self.fulfillment_rate
        self.step_seconds += time.perf_counter() - start_time
        return reward
