"""JSS gym environment (concept-level dispatcher) used by LS, PPO, and PDR.

The environment is registered with gymnasium under id ``jss-v1`` and wraps
the packaged ``source`` submodule (a pinned copy of the JSSEnv benchmark
engine). Most callers can bypass gym.make and instantiate
``JssConceptDispatchEnv`` directly.
"""

from .concept_env import JssConceptDispatchEnv  # noqa: F401
