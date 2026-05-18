import copy
from typing import Union
from .base_node import BaseNode

# Node types, for inheritance to other classes
# Int: integer functions/constants (int return)
# Bool: boolean functions/constants (bool return)
# Statement: expression or terminal action functions (no return)
class IntNode(BaseNode):

    def interpret(self, env) -> int:
        raise Exception('Unimplemented method: interpret')


class BoolNode(BaseNode):

    def interpret(self, env) -> bool:
        raise Exception('Unimplemented method: interpret')


class StatementNode(BaseNode): pass


# Terminal/Non-Terminal types, for inheritance to other classes
class TerminalNode(BaseNode): pass


class OperationNode(BaseNode): pass


# Constants
class ConstBool(BoolNode, TerminalNode):
    
    def __init__(self, value: bool = False):
        super().__init__()
        self.value = value

    def interpret(self, env) -> bool:
        return self.value


class ConstInt(IntNode, TerminalNode):
    
    def __init__(self, value: int = 0):
        super().__init__()
        self.value = value

    def interpret(self, env) -> int:
        return self.value


# Program as an arbitrary node with a single StatementNode child
class Program(BaseNode):

    node_size = 0
    node_depth = 1
    children_types = [StatementNode]

    def run(self, env) -> None:
        assert self.is_complete(), 'Incomplete Program'
        self.children[0].run(env)
    
    def run_generator(self, env):
        assert self.is_complete(), 'Incomplete Program'
        for node in self.get_all_nodes():
            node.reset_state()
        yield from self.children[0].run_generator(env)


# Expressions
class While(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode]

    def reset_state(self):
        self.previous_envs = []

    def run(self, env) -> None:
        while self.children[0].interpret(env):
            # If we have seen this state previously, we're in an infinite loop
            for previous_env in self.previous_envs:
                if env == previous_env:
                    env.crash()
            self.previous_envs.append(copy.deepcopy(env))
            if env.is_crashed(): return     # To avoid infinite loops
            self.children[1].run(env)

    def run_generator(self, env):
        while self.children[0].interpret(env):
            # If we have seen this state previously, we're in an infinite loop
            for previous_env in self.previous_envs:
                if env == previous_env:
                    env.crash()
            self.previous_envs.append(copy.deepcopy(env))
            if env.is_crashed(): return     # To avoid infinite loops
            yield from self.children[1].run_generator(env)


class Repeat(StatementNode, OperationNode):

    node_depth = 1
    children_types = [IntNode, StatementNode]

    def run(self, env) -> None:
        for _ in range(self.children[0].interpret(env)):
            self.children[1].run(env)

    def run_generator(self, env):
        for _ in range(self.children[0].interpret(env)):
            yield from self.children[1].run_generator(env)


class If(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode]

    def run(self, env) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)

    def run_generator(self, env):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)


class ITE(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode, StatementNode]

    def run(self, env) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)
        else:
            self.children[2].run(env)

    def run_generator(self, env):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)
        else:
            yield from self.children[2].run_generator(env)


class Concatenate(StatementNode, OperationNode):

    node_size = 0
    children_types = [StatementNode, StatementNode]

    def run(self, env) -> None:
        self.children[0].run(env)
        self.children[1].run(env)

    def run_generator(self, env):
        yield from self.children[0].run_generator(env)
        yield from self.children[1].run_generator(env)


# Boolean operations
class Not(BoolNode, OperationNode):

    children_types = [BoolNode]
    
    def interpret(self, env) -> bool:
        return not self.children[0].interpret(env)


# Note: And and Or are defined here but are not used in Karel
class And(BoolNode, OperationNode):

    children_types = [BoolNode, BoolNode]
    
    def interpret(self, env) -> bool:
        return self.children[0].interpret(env) and self.children[1].interpret(env)


class Or(BoolNode, OperationNode):

    children_types = [BoolNode, BoolNode]
    
    def interpret(self, env) -> bool:
        return self.children[0].interpret(env) or self.children[1].interpret(env)
    

# For actions available in environment
class Action(StatementNode, TerminalNode):
    
    def run(self, env) -> None:
        if not env.is_crashed():
            env.run_action(self.name)
        
    def run_generator(self, env):
        if not env.is_crashed():
            env.run_action(self.name)
            yield self


class WeightedAction(StatementNode, TerminalNode):
    """Terminal dispatch rule chosen by learnable per-action logits."""

    def __init__(self, action_names=None, logits=None):
        super().__init__("WeightedAction")
        self.action_names = list(action_names or [])
        if logits is None:
            logits = [0.0] * len(self.action_names)
        self.logits = list(logits)

    def selected_action(self):
        if not self.action_names:
            raise ValueError("WeightedAction requires at least one action")
        if len(self.logits) != len(self.action_names):
            raise ValueError("WeightedAction logits/action_names length mismatch")
        best_idx = max(range(len(self.logits)), key=lambda i: self.logits[i])
        return self.action_names[best_idx]

    def run(self, env) -> None:
        if not env.is_crashed():
            env.run_action(self.selected_action())

    def run_generator(self, env):
        if not env.is_crashed():
            env.run_action(self.selected_action())
            yield self


# For features available in environment
class BoolFeature(BoolNode, TerminalNode):
    
    def interpret(self, env) -> bool:
        return env.get_bool_feature(self.name)


class IntFeature(BoolNode, TerminalNode):
    
    def interpret(self, env) -> int:
        return env.get_int_feature(self.name)
