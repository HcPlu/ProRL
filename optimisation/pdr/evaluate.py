"""
Evaluate Priority Dispatching Rules (PDRs) on a JSS instance.

PDRs: FIFO, SPT, MOR, MWR, LOR, Random
Random selects uniformly from {FIFO, SPT, MOR, MWR, LOR} at each action step.

Result storage follows the same convention as ls_JSSEnv.py:
    {logdir}/test/{instance_name}/results.csv

Usage:
    python -m optimisation.pdr.evaluate \
        --logdir <output_dir> \
        --problem_instance_path <path_to_instance> \
        --pdr FIFO
"""

import os
import time
import argparse
import numpy as np
import pandas as pd

from optimisation.jss_env.source.envs.jss_env import JssEnv
from optimisation.jss_env.source.dispatching import (
    FirstInFirstOut,
    ShortestProcessingTime,
    MostOperationsRemaining,
    MostWorkRemaining,
    LeastOperationsRemaining,
)

DETERMINISTIC_RULES = {
    "FIFO": FirstInFirstOut(),
    "SPT": ShortestProcessingTime(),
    "MOR": MostOperationsRemaining(),
    "MWR": MostWorkRemaining(),
    "LOR": LeastOperationsRemaining(),
}

DETERMINISTIC_RULE_LIST = list(DETERMINISTIC_RULES.values())


class RandomPDR:
    """Randomly selects one of {FIFO, SPT, MOR, MWR, LOR} at each decision step."""

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def __call__(self, env):
        rule = DETERMINISTIC_RULE_LIST[self.rng.randint(len(DETERMINISTIC_RULE_LIST))]
        return rule(env)


def run_episode(env, policy_fn):
    """Run one episode with the given policy and return the makespan."""
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy_fn(env)
        obs, reward, done, truncated, _ = env.step(action)
    return env.last_time_step


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PDR on a JSS instance (follows ls_JSSEnv.py storage)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logdir", type=str, required=True,
                        help="Root output directory for this run")
    parser.add_argument("--problem_instance_path", type=str, required=True,
                        help="Path to the JSS instance file")
    parser.add_argument("--pdr", type=str, required=True,
                        choices=["FIFO", "SPT", "MOR", "MWR", "LOR", "Random"],
                        help="PDR to evaluate")
    parser.add_argument("--random_repeats", type=int, default=30,
                        help="Number of repetitions for the Random PDR")
    args = parser.parse_args()

    instance_name = os.path.basename(args.problem_instance_path)

    print(f"Evaluating PDR={args.pdr} on instance={instance_name}")
    print(f"  Instance path: {args.problem_instance_path}")
    print(f"  Logdir:        {args.logdir}")

    env = JssEnv({"instance_path": args.problem_instance_path})
    num_jobs = env.jobs
    num_machines = env.machines

    store_path = os.path.join(args.logdir, "test", instance_name)
    os.makedirs(store_path, exist_ok=True)

    start_time = time.time()

    if args.pdr == "Random":
        rows = []
        for seed in range(args.random_repeats):
            random_pdr = RandomPDR(seed=seed)
            makespan = run_episode(env, random_pdr)
            rows.append({
                "problem": "jsp",
                "instance": instance_name,
                "pdr": "Random",
                "seed": seed,
                "makespan": makespan,
                "num_jobs": num_jobs,
                "num_machines": num_machines,
            })
        elapsed = time.time() - start_time
        for r in rows:
            r["time"] = elapsed

        results_df = pd.DataFrame(rows)
        makespans = results_df["makespan"]
        print(f"  Random ({args.random_repeats} repeats): "
              f"mean={makespans.mean():.1f}  std={makespans.std():.1f}  "
              f"min={makespans.min()}  max={makespans.max()}")
    else:
        rule = DETERMINISTIC_RULES[args.pdr]
        makespan = run_episode(env, rule)
        elapsed = time.time() - start_time

        results_df = pd.DataFrame([{
            "problem": "jsp",
            "instance": instance_name,
            "pdr": args.pdr,
            "seed": 0,
            "makespan": makespan,
            "num_jobs": num_jobs,
            "num_machines": num_machines,
            "time": elapsed,
        }])
        print(f"  {args.pdr}: makespan={makespan}")

    results_path = os.path.join(store_path, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  Results saved to: {results_path}  ({elapsed:.2f}s)")


if __name__ == "__main__":
    main()
