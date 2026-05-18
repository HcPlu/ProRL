"""
Concept-based Job Shop Scheduling Environment.

This module wraps JSSEnv to provide a high-level interface where:
- Actions = Selecting dispatching rules (FIFO, SPT, MOR, MWR, LOR)
- Observations = Concept features (LD, AM, AO, JD, ST)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple
from numba import njit

from optimisation.jss_env.source.envs.jss_env import JssEnv
from optimisation.jss_env.source.dispatching import DISPATCHING_RULES, get_rule

# Fixed order for JIT/Python full vectors; subset observations index into this.
KNOWN_CONCEPTS: Tuple[str, ...] = ("LD", "AM", "AO", "JD", "ST")


@njit(cache=True)
def _compute_concepts_jit(
    jobs: int,
    machines: int,
    legal_actions: np.ndarray,
    todo_time_step_job: np.ndarray,
    machine_ids: np.ndarray,
    processing_times: np.ndarray,
    cumsum_from_end: np.ndarray,
) -> np.ndarray:
    """JIT-compiled concept feature computation."""
    concepts = np.zeros(5, dtype=np.float32)
    
    # Count legal jobs
    n_legal = 0
    for job in range(jobs):
        if legal_actions[job]:
            n_legal += 1
    
    # LD: Machine Load Disbalance
    machine_loads = np.zeros(machines, dtype=np.float64)
    for job in range(jobs):
        for op in range(todo_time_step_job[job], machines):
            machine_id = machine_ids[job, op]
            machine_loads[machine_id] += processing_times[job, op]
    
    max_load = np.max(machine_loads)
    min_load = np.min(machine_loads)
    if max_load > 0:
        concepts[0] = (max_load - min_load) / max_load
    
    # AM: Available Machines
    if n_legal > 0:
        unique_machines = np.zeros(machines, dtype=np.bool_)
        for job in range(jobs):
            if legal_actions[job] and todo_time_step_job[job] < machines:
                machine_needed = machine_ids[job, todo_time_step_job[job]]
                unique_machines[machine_needed] = True
        n_unique = 0
        for m in range(machines):
            if unique_machines[m]:
                n_unique += 1
        concepts[1] = n_unique / machines
    
    # AO: Available Operations
    concepts[2] = n_legal / jobs
    
    # JD: Job Disbalance
    max_remaining = 0.0
    min_remaining = 1e9
    for job in range(jobs):
        if todo_time_step_job[job] < machines:
            remaining = cumsum_from_end[job, todo_time_step_job[job]]
            if remaining > max_remaining:
                max_remaining = remaining
            if remaining < min_remaining:
                min_remaining = remaining
    if max_remaining > 0:
        concepts[3] = (max_remaining - min_remaining) / max_remaining
    
    # ST: Processing Time Spread
    if n_legal > 0:
        max_dur = 0.0
        min_dur = 1e9
        found = False
        for job in range(jobs):
            if legal_actions[job] and todo_time_step_job[job] < machines:
                dur = processing_times[job, todo_time_step_job[job]]
                if dur > max_dur:
                    max_dur = dur
                if dur < min_dur:
                    min_dur = dur
                found = True
        if found and max_dur > 0:
            concepts[4] = (max_dur - min_dur) / max_dur
    
    return concepts


def _compute_concepts_python(
    jobs: int,
    machines: int,
    legal_actions: np.ndarray,
    todo_time_step_job: np.ndarray,
    machine_ids: np.ndarray,
    processing_times: np.ndarray,
    cumsum_from_end: np.ndarray,
) -> np.ndarray:
    """Pure Python version of concept feature computation (no JIT). Used when use_jit=False."""
    concepts = np.zeros(5, dtype=np.float32)
    n_legal = int(np.count_nonzero(legal_actions))
    # LD
    machine_loads = np.zeros(machines, dtype=np.float64)
    for job in range(jobs):
        for op in range(todo_time_step_job[job], machines):
            machine_id = machine_ids[job, op]
            machine_loads[machine_id] += processing_times[job, op]
    max_load = float(np.max(machine_loads))
    min_load = float(np.min(machine_loads))
    if max_load > 0:
        concepts[0] = (max_load - min_load) / max_load
    # AM
    if n_legal > 0:
        unique_machines = np.zeros(machines, dtype=bool)
        for job in range(jobs):
            if legal_actions[job] and todo_time_step_job[job] < machines:
                machine_needed = machine_ids[job, todo_time_step_job[job]]
                unique_machines[machine_needed] = True
        concepts[1] = float(np.count_nonzero(unique_machines)) / machines
    # AO
    concepts[2] = n_legal / jobs
    # JD
    max_remaining = 0.0
    min_remaining = 1e9
    for job in range(jobs):
        if todo_time_step_job[job] < machines:
            remaining = cumsum_from_end[job, todo_time_step_job[job]]
            if remaining > max_remaining:
                max_remaining = remaining
            if remaining < min_remaining:
                min_remaining = remaining
    if max_remaining > 0:
        concepts[3] = (max_remaining - min_remaining) / max_remaining
    # ST
    if n_legal > 0:
        max_dur = 0.0
        min_dur = 1e9
        found = False
        for job in range(jobs):
            if legal_actions[job] and todo_time_step_job[job] < machines:
                dur = processing_times[job, todo_time_step_job[job]]
                if dur > max_dur:
                    max_dur = dur
                if dur < min_dur:
                    min_dur = dur
                found = True
        if found and max_dur > 0:
            concepts[4] = (max_dur - min_dur) / max_dur
    return concepts


class JssConceptDispatchEnv(gym.Env):
    """
    Concept-based wrapper for JSSEnv.
    
    This environment provides:
    - Actions: Discrete selection of dispatching rules
    - Observations: Concept features summarizing the scheduling state
    """
    
    DEFAULT_RULES = ["FIFO", "SPT", "MOR", "MWR", "LOR"]
    DEFAULT_CONCEPTS = list(KNOWN_CONCEPTS)
    
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the concept-based environment.
        
        Args:
            env_config: Configuration dictionary with keys:
                - instance_path: Path to Taillard-style JSS instance file
                - rules: Optional list of dispatching rule names
                - concepts: Optional list of concept names
                - use_jit: If False, disable JIT speedup (pure Python concept computation). Default True.
        """
        super().__init__()
        
        if env_config is None:
            env_config = {}
        
        # Get configuration
        self.use_jit = env_config.get("use_jit", True)
        self.rule_names = list(env_config.get("rules", self.DEFAULT_RULES))
        self.concept_names = list(env_config.get("concepts", self.DEFAULT_CONCEPTS))

        if not self.concept_names:
            raise ValueError("concepts list must be non-empty")
        unknown_c = [c for c in self.concept_names if c not in KNOWN_CONCEPTS]
        if unknown_c:
            raise ValueError(
                f"Unknown concept name(s): {unknown_c}. Allowed: {list(KNOWN_CONCEPTS)}"
            )
        if len(set(self.concept_names)) != len(self.concept_names):
            raise ValueError("concepts must not contain duplicates")
        if not self.rule_names:
            raise ValueError("rules list must be non-empty")
        for r in self.rule_names:
            if r not in DISPATCHING_RULES:
                raise ValueError(
                    f"Unknown dispatching rule '{r}'. Available: {list(DISPATCHING_RULES.keys())}"
                )
        if len(set(self.rule_names)) != len(self.rule_names):
            raise ValueError("rules must not contain duplicates")

        self._concept_indices = np.array(
            [KNOWN_CONCEPTS.index(n) for n in self.concept_names],
            dtype=np.intp,
        )

        # Compatibility attributes (for programEnv interface)
        self.action_space_list = self.rule_names
        self.feature_space = self.concept_names
        self.machine_assignment_space = ["SPT", "EET"]  # Dummy for compatibility
        
        # Create base JSSEnv
        base_config = {"instance_path": env_config.get("instance_path")} if "instance_path" in env_config else None
        self.base_env = JssEnv(base_config)
        
        # Define spaces
        self.action_space = gym.spaces.Discrete(len(self.rule_names))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(len(self.concept_names),), 
            dtype=np.float32
        )
        
        # Pre-extract data from instance_matrix for vectorized computation
        # instance_matrix shape: (jobs, machines, 2) where [:,:,0] = machine_id, [:,:,1] = processing_time
        self._processing_times = self.base_env.instance_matrix[:, :, 1].astype(np.float64)
        self._machine_ids = self.base_env.instance_matrix[:, :, 0].astype(np.int32)
        
        # Pre-compute cumulative sum from end for remaining work calculation
        self._cumsum_from_end = np.flip(
            np.cumsum(np.flip(self._processing_times, axis=1), axis=1), 
            axis=1
        )
        
        # Get dispatching rule objects
        self._rules = [get_rule(name) for name in self.rule_names]
        
        # Store instance info
        self.instance_path = env_config.get("instance_path", "")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment.
        
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.base_env.reset()
        obs = self._compute_concepts()
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step using the selected dispatching rule.
        
        Args:
            action: Index of the dispatching rule to use
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        # Get the dispatching rule
        rule = self._rules[action]
        
        # Use the rule to select the actual job action
        job_action = rule(self.base_env)
        
        # Execute the action in the base environment
        obs, _, done, truncated, info = self.base_env.step(job_action)
        
        # Compute concept features
        concept_obs = self._compute_concepts()
        
        # Reward follows programEnv.py pattern: only negative makespan when done, 0 otherwise
        if done:
            reward = -self.base_env.current_time_step  # Negative makespan
        else:
            reward = 0.0
        
        return concept_obs, reward, done, truncated, info
    
    def _compute_concepts(self) -> np.ndarray:
        """
        Compute concept features from the current state (JIT-compiled implementation).
        
        Features:
        - LD: Machine Load Disbalance
        - AM: Available Machines  
        - AO: Available Operations
        - JD: Job Disbalance
        - ST: Processing Time Spread
        
        Returns:
            Array of concept feature values
        """
        args = (
            self.base_env.jobs,
            self.base_env.machines,
            self.base_env.legal_actions[:-1],  # Exclude no-op
            self.base_env.todo_time_step_job,
            self._machine_ids,
            self._processing_times,
            self._cumsum_from_end,
        )
        if self.use_jit:
            full = _compute_concepts_jit(*args)
        else:
            full = _compute_concepts_python(*args)
        return np.ascontiguousarray(full[self._concept_indices])
    
    def render(self, mode: str = "human"):
        """Render the environment (delegate to base env)."""
        return self.base_env.render(mode)
    
    def get_reward(self) -> float:
        """
        Get the reward following programEnv.py pattern.
        
        Returns:
            - Negative makespan if all operations are scheduled (done)
            - 0.0 otherwise
        """
        # Check if done: no legal actions means all operations are scheduled
        if self.base_env.nb_legal_actions == 0:
            return -self.base_env.current_time_step  # Negative makespan
        return 0.0
    
    @property
    def makespan(self) -> int:
        """Get the current makespan."""
        return self.base_env.current_time_step


if __name__ == "__main__":
    # Simple test
    import time
    
    # Smoke-test default: any JSP instance ships with the repo. Override via
    # argv[1] if you want to load a specific file.
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_instance = os.path.join(_repo, "instances", "jsp", "ft", "ft06")
    instance_path = sys.argv[1] if len(sys.argv) > 1 else default_instance
    
    env = JssConceptDispatchEnv({
        "instance_path": instance_path,
    })
    
    print(f"Action space: {env.action_space_list}")
    print(f"Feature space: {env.feature_space}")
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    done = False
    total_reward = 0.0
    steps = 0
    
    start_time = time.time()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
    
    end_time = time.time()
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward}")
    print(f"Makespan: {env.makespan}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
