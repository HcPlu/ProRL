"""Worker entry point for parallel BO evaluation of esDSL programs."""
import numpy as np


_worker_jit_warmed_up = False


def _warmup_jit_in_worker(collector, warmup_steps: int = 50):
    """Pre-compile Numba JIT functions once per worker process.

    This avoids paying compilation cost on the first real evaluation in each
    forked worker. Skipped when the env was constructed with `use_jit=False`.
    """
    global _worker_jit_warmed_up
    if _worker_jit_warmed_up:
        return
    if not getattr(collector.env, "use_jit", True):
        return
    try:
        env = collector.env
        obs, _ = env.reset()
        for i in range(warmup_steps):
            action = i % len(env.action_space_list)
            obs, _, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
        env.reset()
        _worker_jit_warmed_up = True
    except Exception as e:
        print(f"Worker JIT warmup warning: {e}")


def eval_bayesian_opt_program(
    program,
    optor,
    iterations: int = 100,
    parent_param_merge=None,
    extra_bo_registrations=None,
):
    """Run Bayesian optimisation over `program`'s linear coefficients."""
    _warmup_jit_in_worker(optor.collector)
    optor.parent_param_merge = parent_param_merge or {}
    optor.extra_bo_registrations = extra_bo_registrations or []
    try:
        optor.set_program(program)
        if len(program.get_linear_param_vector()[0]) != 0:
            reward, steps, program, eval_times, telemetry = optor.run(
                iterations=iterations
            )
        else:
            reward, steps, program, eval_times, telemetry = optor.simple_run()
        return reward, steps, program, eval_times, telemetry
    finally:
        optor.parent_param_merge = {}
        optor.extra_bo_registrations = []
        optor.frozen_params = set()
