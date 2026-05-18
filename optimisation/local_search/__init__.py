"""Local search over programmatic policies.

Wraps an esDSL program in a continuous-parameter search space (linear
coefficients + biases of the decision conditions) and optimises it with
Bayesian optimisation or an evolution strategy. Multi-threaded evaluation
helpers live alongside.
"""

from .program import JJSProgram  # noqa: F401
