from ..environment.scheduling import BaseEnvironment
from .base import BaseDSL, _find_close_token
from . import dsl_nodes
from .dsl_nodes import (
    BaseNode, BoolNode, StatementNode,
    TerminalNode, OperationNode,
)
from typing import Union, Generator
from copy import deepcopy
import numpy as np

class LinearBoolExpr(BoolNode, OperationNode):
    node_depth = 0  # No children
    
    def __init__(self, coefficients: list[float] = None, feature_names: list[str] = None):
        super().__init__()
        # First coefficient is the constant term
        self.coefficients = coefficients
        self.feature_names = feature_names
    
    def interpret(self, env) -> bool:
        # Start with the constant term
        result = self.coefficients[0]
        # Add feature terms
        for coef, feat_name in zip(self.coefficients[1:], self.feature_names):
            result += coef * float(env.get_feature(feat_name))
        # Scale-invariant hyperplane: (b + w·x) / ||w|| > 0 — reduces pressure to
        # saturate BO bounds only to sharpen the decision.
        if getattr(env, "linear_bool_scale_invariant", False):
            w = np.asarray(self.coefficients[1:], dtype=float)
            denom = float(np.linalg.norm(w)) + 1e-8
            result = result / denom
        return result > 0

    @staticmethod
    def new(coefficients: list[float], feature_names: list[str]) -> 'LinearBoolExpr':
        return LinearBoolExpr(coefficients, feature_names)

class SchedulingIf(StatementNode, OperationNode):
    node_depth = 1
    children_types = [BoolNode, StatementNode]  # condition and body

    def run(self, env) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)

    def run_generator(self, env):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)

    @staticmethod
    def new(condition: BoolNode, body: StatementNode) -> 'SchedulingIf':
        node = SchedulingIf()
        node.children = [condition, body]
        return node

class SchedulingITE(StatementNode, OperationNode):
    node_depth = 1
    children_types = [BoolNode, StatementNode, StatementNode]  # condition, if_body, else_body

    def __init__(self, name=None):
        super().__init__(name)
        # Instrumentation: [then_branch_hits, else_branch_hits] for dead-branch detection.
        self.branch_counts = [0, 0]

    def reset_branch_counts(self) -> None:
        self.branch_counts = [0, 0]

    def run(self, env) -> None:
        cond = self.children[0].interpret(env)
        if cond:
            self.branch_counts[0] += 1
        else:
            self.branch_counts[1] += 1
        stack = getattr(env, "ite_ancestor_stack", None)
        path_before = list(stack) if stack is not None else []
        if getattr(env, "record_ite_trace", False):
            buf = getattr(env, "ite_trace_buffer", None)
            if buf is not None:
                buf.append((id(self), tuple(path_before), bool(cond)))
        if stack is not None:
            stack.append((id(self), bool(cond)))
        try:
            if cond:
                self.children[1].run(env)
            else:
                self.children[2].run(env)
        finally:
            if stack is not None and len(stack) > 0:
                stack.pop()

    def run_generator(self, env):
        cond = self.children[0].interpret(env)
        if cond:
            self.branch_counts[0] += 1
        else:
            self.branch_counts[1] += 1
        stack = getattr(env, "ite_ancestor_stack", None)
        path_before = list(stack) if stack is not None else []
        if getattr(env, "record_ite_trace", False):
            buf = getattr(env, "ite_trace_buffer", None)
            if buf is not None:
                buf.append((id(self), tuple(path_before), bool(cond)))
        if stack is not None:
            stack.append((id(self), bool(cond)))
        try:
            if cond:
                yield from self.children[1].run_generator(env)
            else:
                yield from self.children[2].run_generator(env)
        finally:
            if stack is not None and len(stack) > 0:
                stack.pop()

    @staticmethod
    def new(condition: BoolNode, if_body: StatementNode, else_body: StatementNode) -> 'SchedulingITE':
        node = SchedulingITE()
        node.children = [condition, if_body, else_body]
        return node

class esDSL(BaseDSL):
    def __init__(
        self,
        heuristic_list: list[str],
        feature_list: list[str],
        weighted_actions: bool = False,
    ):
        self.feature_list = feature_list
        self.heuristic_list = heuristic_list
        self.weighted_actions = bool(weighted_actions)
        action_nodes = [dsl_nodes.Action(h) for h in heuristic_list]
        if self.weighted_actions:
            action_nodes.append(
                dsl_nodes.WeightedAction(
                    action_names=heuristic_list,
                    logits=[0.0] * len(heuristic_list),
                )
            )
        nodes_list = [
            # Control flow nodes
            # SchedulingIf(), 
            SchedulingITE(), 
            # dsl_nodes.Concatenate(), 
            dsl_nodes.Not(), 
            dsl_nodes.And(),
            dsl_nodes.Or(),
            
            # Add LinearBoolExpr with constant term and all features
            LinearBoolExpr(coefficients=[0.0] + [1.0] * len(feature_list), feature_names=feature_list),
            
            # Actions (heuristics)
            *action_nodes,
        ]
        
        super().__init__(nodes_list)

    @property
    def prod_rules(self) -> dict[type[BaseNode], list[list[type[BaseNode]]]]:
        # statements = [SchedulingIf, SchedulingITE, dsl_nodes.Action]
        statements = [ SchedulingITE, dsl_nodes.Action]
        if self.weighted_actions:
            statements.append(dsl_nodes.WeightedAction)
        booleans = [LinearBoolExpr, dsl_nodes.Not, dsl_nodes.And, dsl_nodes.Or]
        # booleans = [LinearBoolExpr,dsl_nodes.Not]
        # booleans = [LinearBoolExpr]
        
        return {
            dsl_nodes.Program: [statements],
            # SchedulingIf: [booleans, statements],
            SchedulingITE: [booleans, statements, statements],
            dsl_nodes.And: [booleans, booleans],
            dsl_nodes.Or: [booleans, booleans],
            dsl_nodes.Not: [booleans],
            # dsl_nodes.Concatenate: [statements, statements]
        }
    
    def get_dsl_nodes_probs(self, node_type):
        # TODO: Ban ITE for now
        if node_type == dsl_nodes.StatementNode:
            probs = {
                # SchedulingIf: 0,
                SchedulingITE: 0.8,
                dsl_nodes.Action: 0.2
            }
            if self.weighted_actions:
                probs = {
                    SchedulingITE: 0.8,
                    dsl_nodes.Action: 0.1,
                    dsl_nodes.WeightedAction: 0.1,
                }
            return probs
        elif node_type == dsl_nodes.BoolNode:
            return {
                dsl_nodes.Not: 0.1,
                dsl_nodes.And: 0.15,
                dsl_nodes.Or: 0.15,
                LinearBoolExpr: 0.6 
            }

    def convert_nodes_to_tokens_list(self, nodes_list: list[dsl_nodes.BaseNode]) -> list[str]:
        tokens_list = ['DEF', 'run', 'm(', 'm)']
        for node in nodes_list:
            if node is None:
                tokens_list += ['<HOLE>']
            
            if isinstance(node, LinearBoolExpr):
                expr = 'LINEAR('
                # Add constant term first
                expr += f'{node.coefficients[0]} + '
                # Add feature terms
                for coef, feat in zip(node.coefficients[1:], node.feature_names):
                    expr += f'{coef}*{feat} + '
                expr = expr[:-3] + ')'  # Remove last ' + ' and close paren
                tokens_list += [expr]
            
            if isinstance(node, dsl_nodes.Action):
                tokens_list += [node.name]

            if isinstance(node, dsl_nodes.WeightedAction):
                tokens_list += [
                    "WACTION("
                    + ",".join(
                        f"{action}:{logit}"
                        for action, logit in zip(node.action_names, node.logits)
                    )
                    + ")"
                ]
            
            if isinstance(node, SchedulingIf):
                tokens_list += ['IF', 'c(', 'c)', 'i(', 'i)']
            if isinstance(node, SchedulingITE):
                tokens_list += ['IFELSE', 'c(', 'c)', 'i(', 'i)', 'ELSE', 'e(', 'e)']
            if isinstance(node, dsl_nodes.Concatenate):
                tokens_list += []
            
            if isinstance(node, dsl_nodes.Not):
                tokens_list += ['not', 'c(', 'c)']
            if isinstance(node, dsl_nodes.And):
                tokens_list += ['and', 'c(', 'c)']
            if isinstance(node, dsl_nodes.Or):
                tokens_list += ['or', 'c(', 'c)']
            
        tokens_list += ['<pad>']
        return list(dict.fromkeys(tokens_list))

    def parse_node_to_str(self, node: dsl_nodes.BaseNode) -> str:
        if node is None:
            return '<HOLE>'
        
        if isinstance(node, LinearBoolExpr):
            expr = 'LINEAR('
            # Add constant term first
            expr += f'{node.coefficients[0]} + '
            # Add feature terms
            for coef, feat in zip(node.coefficients[1:], node.feature_names):
                expr += f'{coef}*{feat} + '
            expr = expr[:-3] + ')'  # Remove last ' + ' and close paren
            return expr
        
        if isinstance(node, dsl_nodes.Action):
            return node.name

        if isinstance(node, dsl_nodes.WeightedAction):
            payload = ",".join(
                f"{action}:{logit}" for action, logit in zip(node.action_names, node.logits)
            )
            return f"WACTION({payload})"
        
        if isinstance(node, dsl_nodes.Program):
            m = self.parse_node_to_str(node.children[0])
            return f'DEF run m( {m} m)'
        
        if isinstance(node, SchedulingIf):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            return f'IF c( {c} c) i( {i} i)'
        if isinstance(node, SchedulingITE):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            e = self.parse_node_to_str(node.children[2])
            return f'IFELSE c( {c} c) i( {i} i) ELSE e( {e} e)'
        if isinstance(node, dsl_nodes.Concatenate):
            s1 = self.parse_node_to_str(node.children[0])
            s2 = self.parse_node_to_str(node.children[1])
            return f'{s1} {s2}'
        
        if isinstance(node, dsl_nodes.Not):
            c = self.parse_node_to_str(node.children[0])
            return f'not c( {c} c)'
        if isinstance(node, dsl_nodes.And):
            c1 = self.parse_node_to_str(node.children[0])
            c2 = self.parse_node_to_str(node.children[1])
            return f'and c( {c1} c) c( {c2} c)'
        if isinstance(node, dsl_nodes.Or):
            c1 = self.parse_node_to_str(node.children[0])
            c2 = self.parse_node_to_str(node.children[1])
            return f'or c( {c1} c) c( {c2} c)'
        
        raise Exception(f'Unknown node type: {type(node)}')

    def parse_str_list_to_node(self, prog_str_list: list[str]) -> dsl_nodes.BaseNode:
        if len(prog_str_list) == 0:
            return None
        
        if prog_str_list[0].startswith('LINEAR('):
            # Parse LINEAR(coef0 + coef1*feat1 + coef2*feat2 + ...)
            expr = ' '.join(prog_str_list)[7:-1] # Remove LINEAR( and )
            terms = expr.split(' + ')
            coefficients = [float(terms[0])]
            feature_names = []


            for term in terms[1:]:
                coef, feat = term.split('*')
                coefficients.append(float(coef))
                feature_names.append(feat)

            assert len(coefficients) == len(feature_names) + 1, 'Invalid LINEAR expression'
            return LinearBoolExpr(coefficients, feature_names)
        
        if prog_str_list[0] in self.actions:
            if len(prog_str_list) > 1:
                s1 = dsl_nodes.Action(prog_str_list[0])
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return dsl_nodes.Action(prog_str_list[0])

        if prog_str_list[0].startswith("WACTION("):
            payload = prog_str_list[0][len("WACTION("):-1]
            action_names = []
            logits = []
            if payload:
                for item in payload.split(","):
                    action, logit = item.split(":", 1)
                    action_names.append(action)
                    logits.append(float(logit))
            node = dsl_nodes.WeightedAction(action_names=action_names, logits=logits)
            if len(prog_str_list) > 1:
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(node, s2)
            return node
        
        if prog_str_list[0] == '<HOLE>':
            if len(prog_str_list) > 1:
                s1 = None
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return None
        
        if prog_str_list[0] == 'DEF':
            assert prog_str_list[1] == 'run', 'Invalid program'
            assert prog_str_list[2] == 'm(', 'Invalid program'
            assert prog_str_list[-1] == 'm)', 'Invalid program'
            m = self.parse_str_list_to_node(prog_str_list[3:-1])
            return dsl_nodes.Program.new(m)
        
        elif prog_str_list[0] == 'IF':
            c_end = _find_close_token(prog_str_list, 'c', 1)
            i_end = _find_close_token(prog_str_list, 'i', c_end+1)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end+2:i_end])
            if i_end == len(prog_str_list) - 1:
                return SchedulingIf.new(c, i)
            else:
                return dsl_nodes.Concatenate.new(
                    SchedulingIf.new(c, i),
                    self.parse_str_list_to_node(prog_str_list[i_end+1:])
                )
        elif prog_str_list[0] == 'IFELSE':
            c_end = _find_close_token(prog_str_list, 'c', 1)
            i_end = _find_close_token(prog_str_list, 'i', c_end+1)
            assert prog_str_list[i_end+1] == 'ELSE', 'Invalid program'
            e_end = _find_close_token(prog_str_list, 'e', i_end+2)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end+2:i_end])
            e = self.parse_str_list_to_node(prog_str_list[i_end+3:e_end])
            if e_end == len(prog_str_list) - 1:
                return SchedulingITE.new(c, i, e)
            else:
                return dsl_nodes.Concatenate.new(
                    SchedulingITE.new(c, i, e),
                    self.parse_str_list_to_node(prog_str_list[e_end+1:])
                )
        
        elif prog_str_list[0] == 'not':
            assert prog_str_list[1] == 'c(', 'Invalid program'
            assert prog_str_list[-1] == 'c)', 'Invalid program'
            c = self.parse_str_list_to_node(prog_str_list[2:-1])
            return dsl_nodes.Not.new(c)
        elif prog_str_list[0] == 'and':
            c1_end = _find_close_token(prog_str_list, 'c', 1)
            assert prog_str_list[c1_end+1] == 'c(', 'Invalid program'
            assert prog_str_list[-1] == 'c)', 'Invalid program'
            c1 = self.parse_str_list_to_node(prog_str_list[2:c1_end])
            c2 = self.parse_str_list_to_node(prog_str_list[c1_end+2:-1])
            return dsl_nodes.And.new(c1, c2)
        elif prog_str_list[0] == 'or':
            c1_end = _find_close_token(prog_str_list, 'c', 1)
            assert prog_str_list[c1_end+1] == 'c(', 'Invalid program'
            assert prog_str_list[-1] == 'c)', 'Invalid program'
            c1 = self.parse_str_list_to_node(prog_str_list[2:c1_end])
            c2 = self.parse_str_list_to_node(prog_str_list[c1_end+2:-1])
            return dsl_nodes.Or.new(c1, c2)
        
        else:
            raise Exception(f'Unrecognized token: {prog_str_list[0]}.')
        
class esDSL_v2(esDSL):
    # boolean is only linear bool expr
    def __init__(
        self,
        heuristic_list: list[str],
        feature_list: list[str],
        weighted_actions: bool = False,
    ):
        super().__init__(heuristic_list, feature_list, weighted_actions=weighted_actions)
        nodes_list = [
            # Control flow nodes
            # SchedulingIf(), 
            SchedulingITE(), 
            # dsl_nodes.Concatenate(), 
            # dsl_nodes.Not(), 
            # dsl_nodes.And(),
            # dsl_nodes.Or(),
            
            # Add LinearBoolExpr with constant term and all features
            LinearBoolExpr(coefficients=[0.0] + [1.0] * len(feature_list), feature_names=feature_list),
            
            # Actions (heuristics)
            *[dsl_nodes.Action(h) for h in heuristic_list],
        ]

    @property
    def prod_rules(self) -> dict[type[BaseNode], list[list[type[BaseNode]]]]:
        statements = [ SchedulingITE, dsl_nodes.Action]
        if self.weighted_actions:
            statements.append(dsl_nodes.WeightedAction)
        booleans = [LinearBoolExpr]
        return {
            dsl_nodes.Program: [statements],
            SchedulingITE: [booleans, statements, statements],

        }
    
    def get_dsl_nodes_probs(self, node_type):
        # TODO: Ban ITE for now
        if node_type == dsl_nodes.StatementNode:
            return {
                # SchedulingIf: 0,
                SchedulingITE: 0.8,
                dsl_nodes.Action: 0.2
            }
        elif node_type == dsl_nodes.BoolNode:
            return {
                LinearBoolExpr: 1
            }
        
        
        
