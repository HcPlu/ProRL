"""ProRL entry point: structural local search + Bayesian optimisation
over an esDSL programmatic dispatching policy on a single JSS instance.

Usage:
    python -m optimisation.local_search.run_ls \
        --problem_instance_path instances/jsp/ft/ft06 \
        --logdir log/prorl_demo --seed 0
"""
from __future__ import annotations

import os, datetime
import time
import numpy as np
from typing import Optional
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter






















from representation.dsl.scheduling import esDSL_v2
from representation.dsl.program_helper import (
    merge_param_dict_parent_into_child,
    linear_param_map_from_program,
)
from representation.space.jjs import JJSProgrammaticSpace
from optimisation.local_search.program import JJSProgram
from optimisation.local_search.ops import bayesian_optimisation
from optimisation.local_search.mthread import eval_bayesian_opt_program
from optimisation.common.rollout import store_data
from optimisation.jss_env.concept_env import JssConceptDispatchEnv

import torch.multiprocessing as mp


mp.set_start_method('spawn', force=True)


def _bo_ls_worker(payload):
    """Multiprocessing entry: Bayesian opt with parent merge + optional shared BO observation."""
    (
        dsl,
        candidate_ast,
        optor,
        iterations,
        parent_ast,
        share_bo_obs,
        parent_reward,
        no_parent_param_merge,
    ) = payload
    pm = {} if no_parent_param_merge else merge_param_dict_parent_into_child(candidate_ast, parent_ast)
    extra = []
    if share_bo_obs and parent_ast is not None:
        pr = float(parent_reward)
        # Skip the shared observation if the parent reward is non-finite
        # (e.g. -inf right after --ls_restart_hard demoted the attractor).
        if np.isfinite(pr):
            extra = [(linear_param_map_from_program(parent_ast), pr)]
    jp = JJSProgram(dsl, candidate_ast)
    return eval_bayesian_opt_program(
        jp,
        optor,
        iterations,
        parent_param_merge=pm,
        extra_bo_registrations=extra,
    )


def estimate_feature_stats(env, max_steps: int = 2000, seed: Optional[int] = None):
    """Univariate mean/std per observation dimension from random rollouts (for z-scoring LINEAR inputs)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    n_feat = len(env.feature_space)
    sums = np.zeros(n_feat, dtype=np.float64)
    sq_sums = np.zeros(n_feat, dtype=np.float64)
    count = 0
    obs, _ = env.reset()
    steps = 0
    while steps < max_steps:
        action_idx = int(rng.randint(0, len(env.action_space_list)))
        obs, _, done, _, _ = env.step(action_idx)
        obs = np.asarray(obs, dtype=np.float64).ravel()
        if obs.shape[0] != n_feat:
            raise ValueError(
                f"estimate_feature_stats: obs length {obs.shape[0]} != len(feature_space) {n_feat}"
            )
        sums += obs
        sq_sums += obs * obs
        count += 1
        steps += 1
        if done:
            obs, _ = env.reset()
    if count == 0:
        return np.zeros(n_feat), np.ones(n_feat)
    mean = sums / count
    var = np.maximum(sq_sums / count - mean * mean, 0.0)
    std = np.sqrt(var) + 1e-8
    return mean, std










# Configure multiprocessing start method
# Use 'fork' on Linux for better performance and to avoid pickling issues with environments
# 'spawn' requires pickling everything which can hang with non-pickleable objects
# import platform
# if platform.system() == 'Linux':
#     try:
#         mp.set_start_method('fork', force=False)
#         print("Using 'fork' start method for multiprocessing")
#     except RuntimeError:
#         # Method already set, try to get current method
#         current_method = mp.get_start_method(allow_none=True)
#         print(f"Multiprocessing start method already set to: {current_method}")
# else:
#     # On Windows/Mac, use 'spawn'
#     try:
#         mp.set_start_method('spawn', force=False)
#         print("Using 'spawn' start method for multiprocessing")
#     except RuntimeError:
#         pass


class mockEnvJSSEnv:
    """Lightweight env-like adapter used by the DSL interpreter during rollouts.

    Wraps a JssConceptDispatchEnv observation so the DSL `get_feature(name)` and
    `run_action(name)` calls can be served without re-allocating the underlying
    env. Optionally z-scores features (stable preset) when feature_mean / feature_std
    are supplied.
    """
    def __init__(
        self,
        obs=None,
        env=None,
        linear_bool_scale_invariant: bool = False,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        self.obs = obs
        self.action_space = env.action_space_list
        self.machine_assignment_space = env.machine_assignment_space
        self.feature_space = env.feature_space
        self.observation_space = len(self.feature_space)
        self.linear_bool_scale_invariant = linear_bool_scale_invariant
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        self.action = None
        if obs is not None:
            self.set_obs(obs)
        else:
            self._feature_list = {f: 0 for f in self.feature_space}

    def set_obs(self, obs):
        self.obs = obs
        self.action = None
        assert len(obs) == len(self.feature_space)
        self._feature_list = {f: obs[i] for i, f in enumerate(self.feature_space)}

    def get_feature(self, feature: str):
        i = self.feature_space.index(feature)
        v = float(self._feature_list[feature])
        if self._feature_mean is not None and self._feature_std is not None:
            v = (v - float(self._feature_mean[i])) / float(self._feature_std[i])
        return v

    def is_crashed(self):
        return False

    def run_action(self, action):
        assert action in self.action_space
        self.action = action

    def get_action(self):
        return self.action

    def get_action_idx(self):
        return self.action_space.index(self.action)


class CollectorJSSEnv:
    """Collector for evaluating an esDSL program on a single JSSEnv instance.

    Each `evaluate_program` call runs `ep` rollouts and returns the per-rollout
    rewards (negative makespan) used as BO fitness signals.
    """
    def __init__(
        self,
        env,
        ep,
        linear_bool_scale_invariant: bool = False,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        self.env = env
        self.instance_name = (
            env.instance_path.split("/")[-1] if env.instance_path else "unknown"
        )
        self.ep = ep
        self.linear_bool_scale_invariant = linear_bool_scale_invariant
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def collect_data(self, program, env=None):
        if env is None:
            env = self.env
        obs, _info = env.reset()
        reward = 0.0
        trajectory = []
        steps = 0
        mockenv = mockEnvJSSEnv(
            None,
            env,
            linear_bool_scale_invariant=self.linear_bool_scale_invariant,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        # Safety cap proportional to instance size to avoid premature truncation
        # on large instances (e.g., 50x20 cscmax/rcmax suites).
        max_steps = (
            getattr(env.base_env, "jobs", 1)
            * getattr(env.base_env, "machines", 1)
            * 10
        )
        while steps < max_steps:
            mockenv.set_obs(obs)
            program.run(mockenv)
            action = mockenv.get_action()
            action_idx = mockenv.get_action_idx()
            obs, r, done, _truncated, info = env.step(action_idx)
            trajectory.append((obs, action, r, done, info))
            reward += r
            steps += 1
            if done:
                break
        return reward, steps, trajectory

    def evaluate_program(self, program):
        trajectories = []
        rewards = []
        steps = []
        for _ in range(self.ep):
            try:
                reward, step, trajectory = self.collect_data(program, env=self.env)
            except Exception:
                continue
            trajectories.append(trajectory)
            rewards.append(reward)
            steps.append(step)
        rewards = [float(r) for r in rewards]
        return np.array(rewards), np.array(steps), trajectories


def save_checkpoint_fn(epoch: int, log_path: str, program, dsl):
    ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}")
    with open(ckpt_path, 'w') as f:
        f.write(dsl.parse_node_to_str(program))

def save_best_fn(log_path: str, program, dsl):
    ckpt_path = os.path.join(log_path, f"best_program")
    with open(ckpt_path, 'w') as f:
        f.write(dsl.parse_node_to_str(program))

class SimpleWriter:
    """Simple writer for logging to files."""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def add_text(self, tag, text):
        with open(os.path.join(self.log_dir, f"{tag}.txt"), 'w') as f:
            f.write(text)


class SimpleLogger:
    """Simple logger that writes to files."""
    def __init__(self, log_path):
        self.writer = SimpleWriter(log_path)
        self._data_log = {}
    
    def write(self, scope_tag, step, data):
        """Write data to CSV file."""
        csv_path = os.path.join(self.writer.log_dir, f"{scope_tag.replace('/', '_')}.csv")
        row = {"step": step, **data}
        df = pd.DataFrame([row])
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)


def create_logger(args, log_name, log_path):
    os.makedirs(log_path, exist_ok=True)
    logger = SimpleLogger(log_path)
    logger.writer.add_text("hparam", str(vars(args)))
    return logger




class LS_v1:
    """ProRL: structural local search + Bayesian optimisation over an esDSL policy.

    Each generation: generate `pop_num` neighbour candidates from the current best,
    evaluate each with successive-halving BO over their linear parameters, and
    promote the best candidate if it improves the incumbent. The per-candidate
    BO budget is `update_times`, split equally across `halving_rounds` halving
    rounds (top 50% survive each round).
    """

    def __init__(
        self,
        dsl,
        p_space,
        feature_list,
        heuristic_list,
        collector,
        optor,
        logger,
        pop_num,
        update_times,
        n_iterations,
        parallel_eval=False,
        seed=1,
        n_jobs=None,
    ):
        self.dsl = dsl
        self.feature_list = feature_list
        self.heuristic_list = heuristic_list
        self.space = p_space
        self.space.set_seed(seed)
        self.collector: CollectorJSSEnv = collector
        self.optor = optor
        self.pop_num = pop_num
        self.update_times = update_times
        self.n_iterations = n_iterations
        self.logger = logger
        self.total_steps = 0
        self.n_jobs = n_jobs if n_jobs is not None else pop_num
        if parallel_eval:
            print(f"Creating multiprocessing pool with {self.n_jobs} processes...")
            self.pool = mp.Pool(processes=self.n_jobs)
            print(f"Pool created successfully with {self.n_jobs} processes")
        else:
            self.pool = None

    def generate_pop(self, program, p=0.5):
        """Generate `pop_num` neighbour candidates from `program`.

        With probability `p` use an action-aware neighbour (replacing a terminal
        heuristic); otherwise use a generic single-node mutation.
        """
        pop = []
        for _ in range(self.pop_num):
            if np.random.random() < p:
                batch = self.space.get_neighbors_with_action(program, k=1)
            else:
                batch = self.space.get_neighbors(program, k=1)
            pop.extend(batch)
        return pop

    @staticmethod
    def _halving_schedule(total_budget, n_rounds):
        """Split `total_budget` BO iterations across `n_rounds` halving rounds."""
        n_rounds = max(1, min(int(n_rounds), int(total_budget)))
        base = total_budget // n_rounds
        schedule = [base] * n_rounds
        for i in range(total_budget - sum(schedule)):
            schedule[-(i % n_rounds) - 1] += 1
        return [max(1, int(s)) for s in schedule]

    def run_parallel(
        self,
        evaluations,
        p=0.5,
        verbose=False,
        successive_halving=True,
        halving_rounds=2,
        share_bo_obs=True,
    ):
        try:
            best_program, _ = self.space.initialize_individual()
            br, _, _ = self.collector.evaluate_program(best_program)
            best_reward = float(np.mean(br)) if len(br) > 0 else float("-inf")

            evaluation_times = 0
            base_update_times = int(self.update_times)
            eval_time_log: list = []
            progress_rows: list = []

            if successive_halving and halving_rounds > 1:
                schedule = self._halving_schedule(base_update_times, halving_rounds)
            else:
                schedule = [base_update_times]

            n = 0
            while evaluation_times < evaluations and n < self.n_iterations:
                candidates = self.generate_pop(best_program, p=p)
                if not candidates:
                    break

                surviving_idx = list(range(len(candidates)))
                pop_results: list = [None] * len(candidates)
                gen_eval_times: list = []
                gen_opt_times: list = []

                for round_idx, round_iters in enumerate(schedule):
                    if not surviving_idx or round_iters < 1:
                        break
                    round_start = time.time()
                    jobs = []
                    for idx in surviving_idx:
                        cand_ast, _meta = candidates[idx]
                        payload = (
                            self.dsl,
                            cand_ast,
                            self.optor,
                            int(round_iters),
                            best_program,
                            bool(share_bo_obs),
                            float(best_reward),
                            False,  # no_parent_param_merge
                        )
                        jobs.append(self.pool.apply_async(_bo_ls_worker, (payload,)))

                    for job_i, job in enumerate(jobs):
                        out = job.get()
                        c_reward, c_steps, c_program = out[0], out[1], out[2]
                        c_eval_times = out[3] if len(out) > 3 else []
                        evaluation_times += int(round_iters)
                        if c_eval_times:
                            gen_eval_times.append(list(c_eval_times))
                        idx = surviving_idx[job_i]
                        if c_program is None or c_reward is None:
                            pop_results[idx] = None
                        else:
                            pop_results[idx] = (c_reward, c_steps, c_program)
                    gen_opt_times.append(time.time() - round_start)

                    # Halving: keep top 50% for the next round
                    if round_idx < len(schedule) - 1:
                        valid = [
                            (idx, pop_results[idx][0])
                            for idx in surviving_idx
                            if pop_results[idx] is not None
                        ]
                        valid.sort(key=lambda x: x[1], reverse=True)
                        keep = max(1, len(valid) // 2)
                        surviving_idx = [idx for idx, _ in valid[:keep]]

                pop = [r for r in pop_results if r is not None]
                log_rewards = [r[0] for r in pop]
                lens = [r[1] for r in pop]

                eval_time_log.append({
                    "gen": n,
                    "individual_eval_times": gen_eval_times,
                    "individual_opt_times_s": gen_opt_times,
                })

                if pop:
                    pop_best = max(pop, key=lambda x: x[0])
                    if pop_best[0] > best_reward:
                        best_program = pop_best[2]
                        best_reward = pop_best[0]
                        save_best_fn(self.logger.writer.log_dir, best_program, self.dsl)

                if verbose:
                    print(f"{n}/{evaluation_times}, best reward: {best_reward}")
                    print(self.dsl.parse_node_to_str(best_program))
                    print(log_rewards)

                store_data(
                    "program_train/env_step",
                    self.logger,
                    self.total_steps,
                    lens,
                    log_rewards,
                )
                save_checkpoint_fn(n, self.logger.writer.log_dir, best_program, self.dsl)
                t_rewards, t_steps, _ = self.collector.evaluate_program(best_program)
                store_data(
                    "program_test/env_step",
                    self.logger,
                    self.total_steps,
                    t_steps,
                    t_rewards,
                )

                best_makespan = (
                    float(-np.mean(t_rewards))
                    if t_rewards is not None and len(t_rewards) > 0
                    else float("nan")
                )
                progress_rows.append({
                    "gen": n,
                    "eval_budget_used": evaluation_times,
                    "eval_budget_total": int(evaluations),
                    "best_so_far_makespan": best_makespan,
                    "population_best_makespan": (
                        float(-max(log_rewards)) if log_rewards else best_makespan
                    ),
                    "population_mean_makespan": (
                        float(-np.mean(log_rewards)) if log_rewards else best_makespan
                    ),
                })

                n += 1

            return best_program, -best_reward, eval_time_log, progress_rows, {}
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool.join()


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description=(
            "ProRL: structural local search + Bayesian optimisation over an "
            "esDSL dispatching policy for a single JSS instance."
        ),
    )
    parser.add_argument("--problem_instance_path", type=str, required=True,
                        help="Path to a JSP instance file (e.g. instances/jsp/ft/ft06).")
    parser.add_argument("--logdir", type=str, default="log/prorl",
                        help="Output directory for results and best-program checkpoints.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the ProRL pipeline.")
    parser.add_argument("--evaluation_times", type=int, default=10000,
                        help="Total ProRL evaluation budget (BO rollouts).")
    parser.add_argument("--update_times", type=int, default=20,
                        help="BO iterations per candidate AST.")
    parser.add_argument("--pop_num", type=int, default=10,
                        help="Number of neighbour candidates per LS generation.")
    parser.add_argument("--n_jobs", type=int, default=8,
                        help="Multiprocessing worker count for parallel BO evaluation.")
    parser.add_argument("--n_iteration", type=int, default=10000,
                        help="Maximum number of LS generations.")
    parser.add_argument("--train_eps", type=int, default=1,
                        help="Number of rollouts per program evaluation.")
    parser.add_argument("--max_tokens", type=int, default=85,
                        help="Maximum AST size in tokens.")
    parser.add_argument("--max_height", type=int, default=4,
                        help="Maximum AST depth.")
    parser.add_argument("--max_sequence", type=int, default=6,
                        help="Maximum length of concatenated terminal sequences.")
    parser.add_argument("--feature_zscore_steps", type=int, default=2000,
                        help="Environment steps for estimating feature mean/std for z-scoring.")
    parser.add_argument("--bo_init_bound", type=float, default=2.0,
                        help="Half-width of the initial BO parameter box.")
    parser.add_argument("--halving_rounds", type=int, default=2,
                        help="Number of successive-halving BO sub-rounds per candidate.")
    parser.add_argument("--no_jit", action="store_true",
                        help="Disable Numba JIT compilation (slower; useful for debugging).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-generation diagnostics.")
    args = parser.parse_args()

    np.random.seed(args.seed)

    env_config = {
        "instance_path": args.problem_instance_path,
        "use_jit": not args.no_jit,
    }
    programEnv = JssConceptDispatchEnv(env_config)
    heuristic_list = programEnv.action_space_list
    feature_list = programEnv.feature_space
    print(f"instance: {programEnv.instance_path}")
    print(f"actions:  {heuristic_list}")
    print(f"features: {feature_list}")

    dsl = esDSL_v2(heuristic_list, feature_list, weighted_actions=False)
    action_probs = {a: 1.0 / len(heuristic_list) for a in heuristic_list}
    p_space = JJSProgrammaticSpace(
        dsl, action_probs, feature_list,
        max_height=args.max_height,
        max_sequence=args.max_sequence,
        max_tokens=args.max_tokens,
        bo_init_bound=args.bo_init_bound,
    )
    p_space.set_seed(args.seed)

    if not args.no_jit:
        print("Warming up Numba JIT compilation...")
        t0 = time.time()
        for _ in range(2):
            obs, _ = programEnv.reset()
            done = False
            step_count = 0
            while not done and step_count < 500:
                obs, _, done, _, _ = programEnv.step(step_count % len(heuristic_list))
                step_count += 1
        print(f"JIT warmup complete in {time.time() - t0:.2f}s")

    print(f"Estimating feature mean/std over {args.feature_zscore_steps} steps...")
    t0 = time.time()
    feature_mean, feature_std = estimate_feature_stats(
        programEnv, max_steps=args.feature_zscore_steps, seed=args.seed
    )
    print(f"Feature stats done in {time.time() - t0:.2f}s")

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join("prorl", str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    logger = create_logger(args, log_name, log_path)

    train_collector = CollectorJSSEnv(
        programEnv,
        args.train_eps,
        linear_bool_scale_invariant=True,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    optor = bayesian_optimisation(train_collector, seed=args.seed)
    optor.base_bo_half_width = float(args.bo_init_bound)
    optor.adaptive_bounds_enabled = True

    ls = LS_v1(
        dsl, p_space, feature_list, heuristic_list,
        train_collector, optor, logger,
        args.pop_num, args.update_times, args.n_iteration,
        parallel_eval=True, seed=args.seed, n_jobs=args.n_jobs,
    )

    start = time.time()
    best_program, best_reward, eval_time_log, progress_rows, _run_meta = ls.run_parallel(
        evaluations=int(args.evaluation_times),
        p=0.1,
        verbose=bool(args.verbose),
        successive_halving=True,
        halving_rounds=int(args.halving_rounds),
        share_bo_obs=True,
    )
    elapsed_time = time.time() - start

    print("best program: ", dsl.parse_node_to_str(best_program))
    print("best reward:  ", best_reward)
    if best_program is not None:
        save_best_fn(logger.writer.log_dir, best_program, dsl)

    final_rewards, _final_steps, _ = train_collector.evaluate_program(best_program)
    if final_rewards is not None and len(final_rewards) > 0:
        makespan_value = float(-np.mean(final_rewards))
    else:
        makespan_value = float("nan")

    instance_name = train_collector.instance_name
    num_jobs = programEnv.base_env.jobs
    num_machines = programEnv.base_env.machines
    store_path = os.path.join(log_path, "test", instance_name)
    os.makedirs(store_path, exist_ok=True)

    total_eval_compute_s = 0.0
    total_eval_wall_clock_s = 0.0
    if eval_time_log:
        gen_rows = []
        all_times = []
        for entry in eval_time_log:
            times_this_gen_flat = []
            for times_s in entry["individual_eval_times"]:
                times_this_gen_flat.extend(times_s)
            if not times_this_gen_flat:
                continue
            per_individual_eval_totals = [sum(times_s) for times_s in entry["individual_eval_times"]]
            max_eval_time_gen_s = max(per_individual_eval_totals) if per_individual_eval_totals else 0.0
            total_eval_wall_clock_s += max_eval_time_gen_s
            gen_rows.append({
                "gen": entry["gen"],
                "total_eval_time_gen_s": sum(times_this_gen_flat),
                "max_eval_time_gen_s": max_eval_time_gen_s,
                "n_eval_calls_gen": len(times_this_gen_flat),
            })
            all_times.extend(times_this_gen_flat)
        total_eval_compute_s = sum(all_times)
        if gen_rows:
            pd.DataFrame(gen_rows).to_csv(
                os.path.join(store_path, "evaluation_times.csv"), index=False
            )

    result_dict = {
        "problem": "jsp",
        "instance": instance_name,
        "makespan": makespan_value,
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "time": elapsed_time,
        "total_eval_compute_s": total_eval_compute_s,
        "total_eval_wall_clock_s": total_eval_wall_clock_s,
    }
    pd.DataFrame([result_dict]).to_csv(os.path.join(store_path, "results.csv"), index=False)
    if progress_rows:
        pd.DataFrame(progress_rows).to_csv(
            os.path.join(store_path, "ls_progress.csv"), index=False
        )
    print("results saved to:", store_path)
