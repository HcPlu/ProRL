from .program import JJSProgram
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
import numpy as np
import time
from copy import deepcopy


def boundary_hit_details(
    params: dict,
    bounds: dict,
    *,
    atol: float = 1e-12,
) -> dict[str, object]:
    """Return exact best-parameter bound hits against active BO bounds."""
    hits: list[dict[str, object]] = []
    param_count = 0
    lower_hit_count = 0
    upper_hit_count = 0
    for name in sorted(params):
        if name not in bounds:
            continue
        lower, upper = bounds[name]
        lower = float(lower)
        upper = float(upper)
        if upper - lower <= 0:
            continue
        value = float(params[name])
        param_count += 1
        side = None
        if abs(value - lower) <= float(atol):
            side = "lower"
            lower_hit_count += 1
        elif abs(value - upper) <= float(atol):
            side = "upper"
            upper_hit_count += 1
        if side is not None:
            hits.append(
                {
                    "name": name,
                    "side": side,
                    "value": value,
                    "lower": lower,
                    "upper": upper,
                }
            )
    hit_names = [hit["name"] for hit in hits]
    hit_count = len(hits)
    return {
        "param_count": int(param_count),
        "hit_count": int(hit_count),
        "hit_fraction": float(hit_count / param_count) if param_count else 0.0,
        "lower_hit_count": int(lower_hit_count),
        "upper_hit_count": int(upper_hit_count),
        "hit_param_names": hit_names,
        "hits": hits,
    }


def expanded_bounds_for_boundary_hits(
    bounds: dict,
    details: dict,
    *,
    multiplier: float = 2.0,
    cap: float = 8.0,
) -> dict:
    """Expand only dimensions that hit a bound, preserving all other bounds."""
    expanded = {name: (float(lo), float(hi)) for name, (lo, hi) in bounds.items()}
    hit_names = set(details.get("hit_param_names", []) if isinstance(details, dict) else [])
    for name in hit_names:
        if name not in expanded:
            continue
        lower, upper = expanded[name]
        center = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        if half_width <= 0:
            continue
        new_half_width = min(float(cap), half_width * float(multiplier))
        expanded[name] = (center - new_half_width, center + new_half_width)
    return expanded


def split_boundary_refit_budget(total_iterations: int, phase1_frac: float) -> tuple[int, int]:
    """Split BO suggestions into fixed-budget base and refit phases."""
    total = max(0, int(total_iterations))
    if total == 0:
        return 0, 0
    frac = min(1.0, max(0.0, float(phase1_frac)))
    phase1 = int(round(total * frac))
    phase1 = max(1, min(total, phase1))
    return phase1, total - phase1




# Flatten and encode the model's parameters

# Decode the vector and rebuild the model



class bayesian_optimisation:
    def __init__(self,  collector, n_iter=100, n_init=10, acq='ucb', kappa=2.5, xi=0.05, seed=None):
        self.collector = collector
        self.n_iter = n_iter
        self.n_init = n_init
        self.acq = acq
        self.kappa = kappa
        self.xi = xi
        self.seed = seed
        self.optimizer = None
        self.debug = False
        # Optional: merge parent program coefficients into get_init_points (path-aligned keys).
        self.parent_param_merge = {}
        # Per-parameter bound half-width multiplier (adaptive bounds); cap applied in set_bounds.
        self._per_param_expansion = {}
        self.base_bo_half_width = 2.0
        self.adaptive_bounds_enabled = False
        self.adaptive_bound_cap = 16.0
        # Params frozen (near-constant bounds) e.g. dead P-dead subtree coefficients.
        self.frozen_params = set()
        # Space-filling BO init (Phase A1). bo_init_points=1 => legacy single-probe.
        self.bo_init_points = 1
        self.bo_init_design = "random"  # "random" | "sobol" | "lhs"
        self.bo_suggest_n_random = None
        self.bo_suggest_n_l_bfgs_b = None
        self.boundary_refit_enabled = False
        self.boundary_refit_phase1_frac = 0.7
        self.boundary_refit_multiplier = 2.0
        self.boundary_refit_cap = 8.0
    
    def set_program(self, program):
        assert isinstance(program, JJSProgram)
        self.program = program

    def simple_run(self):
        # if self.optimizer is None:
        #     raise ValueError("Optimizer not initialized")
        # self.optimizer.maximize(init_points=self.n_init, n_iter=self.n_iter, acq=self.acq, kappa=self.kappa, xi=self.xi)
        # return self.optimizer.max['params']

        self._eval_times = []
        t0 = time.time()
        e_rewards, e_steps, _ = self.collector.evaluate_program(self.program.get_node_program())
        self._eval_times.append(time.time() - t0)
        if len(e_rewards) == 0:
            return None, None, None, [], {"pct_dims_bound_hit_mean": 0.0, "dead_branch_count": 0}
        self._run_telemetry = {
            "pct_dims_bound_hit_mean": 0.0,
            "dead_branch_count": int(
                getattr(self.collector, "_last_dead_branch_count", 0)
            ),
            "pre_bo_reward": float(e_rewards.mean()),
        }
        return (
            e_rewards.mean(),
            e_steps.sum(),
            self.program.get_node_program(),
            list(self._eval_times),
            self._run_telemetry,
        )

    def run(self,iterations = 100,seed=1):
        self._eval_times = []
        self._telemetry_bound_fracs = []
        _suggest_times = []
        _register_times = []

        t_start = time.time()
        acq = acquisition.UpperConfidenceBound(kappa=self.kappa)
        # acq = acquisition.ExpectedImprovement(xi=self.xi)
        bounds = self.set_bounds()
        optimizer = BayesianOptimization(
            f=None,
            acquisition_function=acq,
            pbounds=bounds,
            verbose=0,
            random_state=self.seed,
            allow_duplicate_points=True
    )
        t_init = time.time() - t_start

        init_pts = self.get_init_points()
        extra = getattr(self, "extra_bo_registrations", None) or []
        for reg_params, reg_target in extra:
            merged = {**init_pts, **{k: reg_params[k] for k in init_pts if k in reg_params}}
            optimizer.probe(merged, lazy=True)
            optimizer.register(params=merged, target=reg_target)

        optimizer.probe(init_pts,lazy=True)
        t0 = time.time()
        init_target = self.b_function(init_pts)[0]
        if init_target is None:
            return (
                None,
                None,
                None,
                [],
                {"pct_dims_bound_hit_mean": 0.0, "dead_branch_count": 0},
            )
        optimizer.register(params=init_pts, target=init_target)
        t_first_eval = time.time() - t0

        # Phase A1: optional space-filling init points in addition to the
        # anchor probe. Budget is kept neutral by deducting from UCB iters.
        extra_init = max(0, int(getattr(self, "bo_init_points", 1)) - 1)
        extra_init = min(extra_init, max(0, iterations))
        total_steps = 0
        if extra_init > 0:
            design = getattr(self, "bo_init_design", "random")
            space_pts = self._space_filling_init_points(
                extra_init, design=design, seed=self.seed
            )
            for pt in space_pts:
                tgt, e_steps = self.b_function(pt)
                if tgt is None:
                    return (
                        None,
                        None,
                        None,
                        [],
                        {"pct_dims_bound_hit_mean": 0.0, "dead_branch_count": 0},
                    )
                optimizer.register(params=pt, target=tgt)
                total_steps += e_steps
            iterations = iterations - extra_init

        def _run_suggestions(n_suggestions: int) -> tuple[bool, int]:
            nonlocal total_steps
            for i in range(int(n_suggestions)):
                t0 = time.time()
                next_point = self._suggest_next_point(optimizer)
                _suggest_times.append(time.time() - t0)

                target, e_steps = self.b_function(next_point)
                if target is None:
                    return False, total_steps

                t0 = time.time()
                optimizer.register(params=next_point, target=target)
                _register_times.append(time.time() - t0)

                total_steps += e_steps
                if self.debug:
                    print(
                        "iter: ",
                        i,
                        "reward: ",
                        target,
                        "total_steps: ",
                        total_steps,
                        optimizer.max["target"],
                    )
            return True, total_steps

        phase1_iters = int(iterations)
        phase2_iters = 0
        boundary_refit_details = boundary_hit_details(optimizer.max["params"], bounds)
        expanded_param_names: list[str] = []
        if getattr(self, "boundary_refit_enabled", False):
            phase1_iters, phase2_iters = split_boundary_refit_budget(
                int(iterations),
                float(getattr(self, "boundary_refit_phase1_frac", 0.7)),
            )

        ok, total_steps = _run_suggestions(phase1_iters)
        if not ok:
            return (
                None,
                None,
                None,
                [],
                {"pct_dims_bound_hit_mean": 0.0, "dead_branch_count": 0},
            )

        if getattr(self, "boundary_refit_enabled", False):
            boundary_refit_details = boundary_hit_details(optimizer.max["params"], bounds)
            expanded_bounds = expanded_bounds_for_boundary_hits(
                bounds,
                boundary_refit_details,
                multiplier=float(getattr(self, "boundary_refit_multiplier", 2.0)),
                cap=float(getattr(self, "boundary_refit_cap", 8.0)),
            )
            expanded_param_names = [
                name
                for name in sorted(expanded_bounds)
                if tuple(expanded_bounds[name]) != tuple(bounds.get(name, (None, None)))
            ]
            if expanded_param_names and phase2_iters > 0:
                optimizer.set_bounds(expanded_bounds)
                self._reference_bounds = dict(expanded_bounds)
                bounds = expanded_bounds

        ok, total_steps = _run_suggestions(phase2_iters)
        if not ok:
            return (
                None,
                None,
                None,
                [],
                {"pct_dims_bound_hit_mean": 0.0, "dead_branch_count": 0},
            )
        
        print(f"BO timing: init={t_init:.3f}s, first_eval={t_first_eval:.3f}s, "
              f"avg_suggest={np.mean(_suggest_times):.3f}s, avg_register={np.mean(_register_times):.4f}s, "
              f"total_suggest={sum(_suggest_times):.2f}s, total_eval={sum(self._eval_times):.2f}s")
        
        reward = optimizer.max['target']
        params = optimizer.max['params']

        self.program.set_linear_param_vector_params(params, self.program._feature_lists)
        self._update_adaptive_expansion_from_best(optimizer)

        fracs = getattr(self, "_telemetry_bound_fracs", None) or []
        self._run_telemetry = {
            "pct_dims_bound_hit_mean": float(np.mean(fracs)) if fracs else 0.0,
            "dead_branch_count": int(
                getattr(self.collector, "_last_dead_branch_count", 0)
            ),
            "pre_bo_reward": float(init_target),
            "boundary_refit_enabled": bool(getattr(self, "boundary_refit_enabled", False)),
            "boundary_refit_phase1_iters": int(phase1_iters),
            "boundary_refit_phase2_iters": int(phase2_iters),
            "boundary_hit_count": int(boundary_refit_details.get("hit_count", 0)),
            "boundary_hit_fraction": float(boundary_refit_details.get("hit_fraction", 0.0)),
            "boundary_lower_hit_count": int(boundary_refit_details.get("lower_hit_count", 0)),
            "boundary_upper_hit_count": int(boundary_refit_details.get("upper_hit_count", 0)),
            "boundary_hit_param_names": list(
                boundary_refit_details.get("hit_param_names", [])
            ),
            "boundary_expanded_param_names": expanded_param_names,
            "boundary_active_bounds": dict(bounds),
        }

        return (
            reward,
            total_steps,
            self.program.get_node_program(),
            list(self._eval_times),
            self._run_telemetry,
        )

    def _suggest_next_point(self, optimizer):
        n_random = getattr(self, "bo_suggest_n_random", None)
        n_l_bfgs_b = getattr(self, "bo_suggest_n_l_bfgs_b", None)
        if n_random is None and n_l_bfgs_b is None:
            return optimizer.suggest()

        kwargs = {"gp": optimizer._gp, "target_space": optimizer._space, "fit_gp": True}
        if n_random is not None:
            kwargs["n_random"] = int(n_random)
        if n_l_bfgs_b is not None:
            kwargs["n_l_bfgs_b"] = int(n_l_bfgs_b)
        suggestion = optimizer._acquisition_function.suggest(**kwargs)
        return optimizer._space.array_to_params(suggestion)

    def _update_adaptive_expansion_from_best(self, optimizer):
        if not getattr(self, "adaptive_bounds_enabled", False):
            return
        best = optimizer.max.get("params")
        if not best:
            return
        for name, val in best.items():
            lo, hi = self._reference_bounds.get(name, (-self.base_bo_half_width, self.base_bo_half_width))
            span = hi - lo
            if span <= 0:
                continue
            if abs(val - hi) < 0.05 * span or abs(val - lo) < 0.05 * span:
                cur = self._per_param_expansion.get(name, 1.0)
                self._per_param_expansion[name] = min(cur * 2.0, self.adaptive_bound_cap / self.base_bo_half_width)

    def get_init_points(self):
        _parameters, _feature_lists = self.program.get_linear_param_vector()
        param_name_vector = []

        for params in self.program.get_linear_param_vector()[1]:
            param_name_vector += params

        init_points = {}
        for i,param in enumerate(param_name_vector):
            init_points[param] = _parameters[i]
        merge = getattr(self, "parent_param_merge", None) or {}
        for k in init_points:
            if k in merge:
                init_points[k] = merge[k]
        return init_points

    def _space_filling_init_points(self, k, design="sobol", seed=None):
        """Generate k additional init points inside self._reference_bounds via a
        space-filling design. Falls back to uniform random if SciPy QMC is
        unavailable. Output: list of dicts (keys = param names)."""
        ref = getattr(self, "_reference_bounds", None) or {}
        if not ref or k <= 0:
            return []
        names = list(ref.keys())
        lows = np.array([ref[n][0] for n in names], dtype=np.float64)
        highs = np.array([ref[n][1] for n in names], dtype=np.float64)
        d = len(names)
        rng_seed = None
        if seed is not None:
            rng_seed = int(seed) & 0x7FFFFFFF
        try:
            from scipy.stats import qmc
            if design == "lhs":
                sampler = qmc.LatinHypercube(d=d, seed=rng_seed)
                unit = sampler.random(n=k)
            else:  # "sobol" (default) or unknown -> sobol
                sampler = qmc.Sobol(d=d, scramble=True, seed=rng_seed)
                unit = sampler.random(n=k)
        except Exception:
            rng = np.random.RandomState(rng_seed)
            unit = rng.uniform(size=(k, d))
        scaled = lows + unit * (highs - lows)
        pts = []
        for row in scaled:
            pts.append({name: float(val) for name, val in zip(names, row)})
        return pts
    
    def _pct_params_near_bound(self, params: dict) -> float:
        ref = getattr(self, "_reference_bounds", None) or {}
        if not ref:
            return 0.0
        hits = 0
        n = 0
        for k, v in params.items():
            if k not in ref:
                continue
            lo, hi = ref[k]
            span = hi - lo
            if span <= 1e-12:
                continue
            n += 1
            if abs(float(v) - lo) <= 0.05 * span or abs(float(v) - hi) <= 0.05 * span:
                hits += 1
        return hits / max(1, n)

    def b_function(self, params):
        self.program.set_linear_param_vector_params(params, self.program._feature_lists)
        t0 = time.time()
        e_rewards, e_steps, _ = self.collector.evaluate_program(self.program.get_node_program())
        eval_time = time.time() - t0
        self._eval_times.append(eval_time)
        if getattr(self, "_telemetry_bound_fracs", None) is not None:
            self._telemetry_bound_fracs.append(self._pct_params_near_bound(params))
        # Debug print removed to reduce I/O overhead during BO iterations
        # print(f"Evaluation time: {eval_time}")
        if len(e_rewards) == 0:
            return None, None
        return e_rewards.mean(), e_steps.sum()
    
    def set_bounds(self):
        bounds = {}
        param_name_vector = []

        for params in self.program.get_linear_param_vector()[1]:
            param_name_vector += params

        self._reference_bounds = {}
        frozen = getattr(self, "frozen_params", set()) or set()
        init_pts = {}
        _parameters, _ = self.program.get_linear_param_vector()
        idx = 0
        for params in self.program.get_linear_param_vector()[1]:
            for p in params:
                init_pts[p] = _parameters[idx]
                idx += 1

        for param in param_name_vector:
            half = self.base_bo_half_width * self._per_param_expansion.get(param, 1.0)
            half = min(half, getattr(self, "adaptive_bound_cap", 16.0))
            lo, hi = (-half, half)
            if param in frozen:
                v = init_pts.get(param, 0.0)
                lo, hi = (v - 1e-9, v + 1e-9)
            bounds[param] = (lo, hi)
            self._reference_bounds[param] = (lo, hi)
        return bounds
    
    def set_optimizer(self, program, env):
        self.program = program
        self.env = env
        bounds = self.set_bounds()
        self.optimizer = BayesianOptimization(f=self.b_function, pbounds=bounds, verbose=0, random_state=self.seed)




        

