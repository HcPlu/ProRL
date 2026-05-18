from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Union, Callable

import numpy as np

class BaseEnvironment(ABC):
    def __init__(self, actions: dict[str, Callable], features: dict[str, Callable], state_shape: tuple[int, ...],
                 initial_state: Union[Any, None] = None, max_calls: int = 5000, max_steps: int = 1000):
        self.actions = actions
        self.actions_list = list(actions.keys())
        self.features = features
        self.features_list = list(features.keys())
        self.max_calls = max_calls
        self.max_steps = max_steps
        self.state_shape = state_shape
        self.num_calls: int = 0
        self.num_steps: int = 0
        self.crashed: bool = False
        self.current_action_str = None
        if initial_state is not None:
            self.set_state(initial_state)
        else:
            self.set_state(self.default_state())

    def is_crashed(self) -> bool:
        return self.crashed
    
    def crash(self):
        self.crashed = True

    
    def get_feature(self, feature: str):
        self.num_calls += 1
        if self.num_calls > self.max_calls:
            self.crashed = True
        return self.features[feature]()


    def run_action(self, action: str):
        self.num_calls += 1
        self.num_steps += 1
        self.current_action_str = action
        if self.num_steps > self.max_steps:
            self.crashed = True
        if self.num_calls > self.max_calls:
            self.crashed = True
        self.actions[action]()
        
    def run_action_index(self, action_index: int):
        self.num_calls += 1
        self.num_steps += 1
        if self.num_steps > self.max_steps:
            self.crashed = True
        if self.num_calls > self.max_calls:
            self.crashed = True
        self.actions[self.actions_list[action_index]]()

    @abstractmethod
    def default_state(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self):
        pass
    
    @abstractmethod
    def __eq__(self, other: BaseEnvironment) -> bool:
        pass

    @classmethod
    @abstractmethod
    def from_string(cls, state_str: str):
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def to_image(self) -> np.ndarray:
        pass
