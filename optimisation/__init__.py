"""Optimisation layer.

Algorithms that search over (or fit parameters of) policies for scheduling:

  optimisation/local_search/  - Bayesian / evolution-strategy local search
                                over programmatic policies (esDSL / esDSL_v2)
  optimisation/ppo/           - PPO baseline (learned neural policy)
  optimisation/pdr/           - priority-dispatching-rule baselines
  optimisation/common/        - rollout / evaluation utilities shared
                                across methods

Entry-point scripts live inside each sub-package (run_ls.py, run_ppo.py,
evaluate.py). The representation/ layer is the dependency — everything
here consumes programmatic policies defined there.
"""
