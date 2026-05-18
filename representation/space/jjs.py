from __future__ import annotations
import copy
import numpy as np

from ..dsl import dsl_nodes
from ..dsl.scheduling import esDSL, SchedulingIf, SchedulingITE, LinearBoolExpr
from .base import BaseSearchSpace
def get_max_height(program: dsl_nodes.Program) -> int:
    """Calculates the maximum height of an input program AST

    Args:
        program (dsl_nodes.Program): Input program

    Returns:
        int: Maximum height of AST
    """
    height = 0
    for child in program.children:
        if child is not None:
            height = max(height, get_max_height(child))
    return height + program.node_depth

def get_node_current_height(node: dsl_nodes.BaseNode) -> int:
    """Calculates the current height of an input node in a program AST

    Args:
        node (dsl_nodes.BaseNode): Input node

    Returns:
        int: Current height of node
    """
    height = node.node_depth
    while not isinstance(node, dsl_nodes.Program):
        height += node.parent.node_depth
        node = node.parent
    return height

def get_max_sequence(program: dsl_nodes.Program, _current_sequence = 1, _max_sequence = 0) -> int:
    """Returns the length of maximum sequence of Concatenate nodes in an input program

    Args:
        program (dsl_nodes.Program): Input program

    Returns:
        int: Length of maximum sequence of Concatenate nodes
    """
    if isinstance(program, dsl_nodes.Concatenate):
        _current_sequence += 1
    else:
        _current_sequence = 1
    _max_sequence = max(_max_sequence, _current_sequence)
    for child in program.children:
        _max_sequence = max(_max_sequence, get_max_sequence(child, _current_sequence, _max_sequence))
    return _max_sequence

def get_node_current_sequence(node: dsl_nodes.BaseNode) -> int:
    """Returns the length of the current sequence of Concatenate nodes in an input program

    Args:
        node (dsl_nodes.BaseNode): Input node

    Returns:
        int: Length of current sequence of Concatenate nodes
    """
    current_sequence = 1
    while isinstance(node, dsl_nodes.Concatenate):
        current_sequence += 1
        node = node.parent
    return current_sequence


class JJSProgrammaticSpace(BaseSearchSpace):
    
    def __init__(
        self,
        dsl: esDSL,
        action_probs: dict[str, float],
        features: list[str],
        max_height: int = 4,
        max_sequence: int = 6,
        max_tokens: int = 300,
        min_tokens: int = 10,
        bo_init_bound: float = 2.0,
        n_mutations: int = 1,
        depth_bias_alpha: float = 0.0,
        mutation_guidance_mode: str = "none",
    ) -> None:
        super().__init__(dsl)
        self.action_probs = action_probs
        self.features = features
        self.max_height = max_height
        self.max_sequence = max_sequence
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        # Match BO linear bounds in optimisation/local_search/ops.py (default ±2).
        self.bo_init_bound = float(bo_init_bound)
        # Number of single-node mutations per neighbor (D4 multi-node mutation).
        self.n_mutations = max(1, int(n_mutations))
        # Iter 10: bias mutation-target sampling so shallow nodes are
        # preferred. weight ∝ (1 + depth) ^ (-alpha). 0.0 = uniform (legacy).
        self.depth_bias_alpha = float(depth_bias_alpha)
        # Iter 15 per-candidate α ensemble: list of α values cycled over
        # pop_num candidates per generation. None = single scalar (legacy).
        self.depth_bias_alphas: list[float] | None = None
        self.mutation_guidance_mode = str(mutation_guidance_mode)
        self.last_neighbor_metadata: list[dict[str, object]] = []
    
    def get_height(self, program: dsl_nodes.Program) -> int:
        return get_max_height(program)
    
    def get_sequence(self, program: dsl_nodes.Program) -> int:
        return get_max_sequence(program)
    
    def get_tokens_num(self, program: dsl_nodes.Program) -> int:
        return len(self.dsl.parse_node_to_str(program).split(" "))
    
    def _fill_children(self, node: dsl_nodes.BaseNode,
                          current_height: int = 1, current_sequence: int = 0,
                          max_height: int = 4, max_sequence: int = 6) -> None:
        """Recursively fills the children of a program node

        Args:
            node (dsl_nodes.BaseNode): Input node
            current_height (int, optional): Height of current element, for recursion. Defaults to 1.
            current_sequence (int, optional): Sequence of current element, for recursion. Defaults to 0.
            max_height (int, optional): Maximum allowed AST height. Defaults to 4.
            max_sequence (int, optional): Maximum allowed Concatenate sequence. Defaults to 6.
        """
        node_prod_rules = self.dsl.prod_rules[type(node)]
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = self.dsl.get_dsl_nodes_probs(child_type)
            for child_type in child_probs:
                if child_type not in node_prod_rules[i]:
                    child_probs[child_type] = 0.
                if current_height >= max_height and child_type.get_node_depth() > 0:
                    child_probs[child_type] = 0.
            if isinstance(node, dsl_nodes.Concatenate) and current_sequence + 1 >= max_sequence:
                if dsl_nodes.Concatenate in child_probs:
                    child_probs[dsl_nodes.Concatenate] = 0.
            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if isinstance(node, dsl_nodes.Concatenate):
                    self._fill_children(child_instance, current_height + child.get_node_depth(),
                                        current_sequence + 1, max_height, max_sequence)
                else:
                    self._fill_children(child_instance, current_height + child.get_node_depth(),
                                        1, max_height, max_sequence)
            
            # TODO: Terminal nodes
            elif isinstance(child_instance, dsl_nodes.Action):
                #randomly generate action
                child_instance.name = self.np_rng.choice(list(self.action_probs.keys()),
                                                         p=list(self.action_probs.values()))
            elif isinstance(child_instance, LinearBoolExpr):
                child_instance.feature_names = self.features
                b = self.bo_init_bound
                const = round(float(self.np_rng.normal(0.0, 0.5)), 4)
                rest = [
                    round(float(self.np_rng.uniform(-b, b)), 4)
                    for _ in range(len(child_instance.feature_names))
                ]
                child_instance.coefficients = [const] + rest
      
            node.children[i] = child_instance
            child_instance.parent = node

    def initialize_individual(self) -> tuple[dsl_nodes.Program, dsl_nodes.Program]:
        """Initializes individual using probabilistic DSL

        Returns:
            tuple[dsl_nodes.Program, dsl_nodes.Program]: Individual as tuple of
            program (individual) and program (decoding)
        """
        accepted = False
        while not accepted:
            program = dsl_nodes.Program()
            self._fill_children(program, max_height=self.max_height, max_sequence=self.max_sequence)
            prog_str = self.dsl.parse_node_to_str(program)
            # print(len(prog_str.split(" ")),prog_str,self.max_height,self.max_sequence)
            # TODO: Add length constraint\
            accepted = get_max_height(program) <= self.max_height and get_max_sequence(program) <= self.max_sequence and\
                  len(prog_str.split(" ")) <= self.max_tokens and len(prog_str.split(" ")) >= self.min_tokens
        self._merge_action_nodes(program)
        return program, program
    
    def _mutate_node(self, node_to_mutate: dsl_nodes.BaseNode) -> None:
        """Mutates a node in a program by replacing it with a random node of the same type

        Args:
            node_to_mutate (dsl_nodes.BaseNode): Program node to mutate
        """
        for i, child in enumerate(node_to_mutate.parent.children):
            if child == node_to_mutate:
                child_type = node_to_mutate.parent.children_types[i]
                node_prod_rules = self.dsl.prod_rules[type(node_to_mutate.parent)]
                child_probs = self.dsl.get_dsl_nodes_probs(child_type)
                for child_type in child_probs:
                    if child_type not in node_prod_rules[i]:
                        child_probs[child_type] = 0.
                
                p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
                child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
                child_instance = child()
                if child.get_number_children() > 0:
                    curr_seq = get_node_current_sequence(node_to_mutate)
                    if child_type == dsl_nodes.Concatenate:
                        curr_seq += 1
                    else:
                        curr_seq = 1
                    curr_height = get_node_current_height(node_to_mutate) + child.get_node_depth()
                    self._fill_children(child_instance, current_height=curr_height, current_sequence=curr_seq,
                        max_height=self.max_height, max_sequence=self.max_sequence)
                # TODO: Terminal nodes
                elif isinstance(child_instance, dsl_nodes.Action):
                    #randomly generate action
                    child_instance.name = self.np_rng.choice(list(self.action_probs.keys()),
                                                            p=list(self.action_probs.values()))
                elif isinstance(child_instance, LinearBoolExpr):
                    b = self.bo_init_bound
                    child_instance.feature_names = self.features
                    const = round(float(self.np_rng.normal(0.0, 0.5)), 4)
                    rest = [
                        round(float(self.np_rng.uniform(-b, b)), 4)
                        for _ in range(len(child_instance.feature_names))
                    ]
                    child_instance.coefficients = [const] + rest
                    
                node_to_mutate.parent.children[i] = child_instance
                child_instance.parent = node_to_mutate.parent
    
    def _mutate_node_with_action(self, nodes: dsl_nodes.BaseNode,prob: float = 0.5) -> None:
        """Mutates a node in a program by replacing it with a random action node

        Args:
            node_to_mutate (dsl_nodes.BaseNode): Program node to mutate
        """
        action_nodes = [node for node in nodes if isinstance(node, dsl_nodes.Action)]
        if len(action_nodes) == 0:
            return
        mutated_num = int(len(action_nodes) * prob) if len(action_nodes) * prob > 1 else 1
        #randomly select
        mutated_action_nodes = self.np_rng.choice(action_nodes, size=mutated_num, replace=False)
        for action_node in mutated_action_nodes:
            action_choice = self.np_rng.choice(list(self.action_probs.keys()),
                                                            p=list(self.action_probs.values()))
            # print(action_node.name,action_choice)
            action_node.name = action_choice

    def _node_guidance_weight(self, node: dsl_nodes.BaseNode) -> float:
        if self.mutation_guidance_mode == "none":
            return 1.0
        if self.mutation_guidance_mode == "shallow_action_bias":
            if isinstance(node, dsl_nodes.Action):
                return 3.0
            if isinstance(node, SchedulingITE):
                return 2.5
            if isinstance(node, LinearBoolExpr):
                return 2.0
            return 1.0
        return 1.0

    def _sample_mutation_node(
        self,
        nodes: list[dsl_nodes.BaseNode],
        *,
        alpha: float,
    ) -> dsl_nodes.BaseNode:
        if not nodes:
            raise ValueError("Cannot sample a mutation node from an empty list")
        weights = np.ones(len(nodes), dtype=float)
        if alpha > 0.0:
            depths = np.array([n.get_depth() for n in nodes], dtype=float)
            weights *= 1.0 / np.power(1.0 + depths, alpha)
        if self.mutation_guidance_mode != "none":
            weights *= np.array([self._node_guidance_weight(n) for n in nodes], dtype=float)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            return self.np_rng.choice(nodes)
        weights = weights / total
        return self.np_rng.choice(nodes, p=weights)

    def _merge_action_nodes(self, node: dsl_nodes.BaseNode) -> None:
        """Merges if-else nodes where both branches have the same action.
        
        Args:
            node (dsl_nodes.BaseNode): The node to process
        """
        if isinstance(node, SchedulingITE):
            if isinstance(node.children[1], dsl_nodes.Action) and isinstance(node.children[2], dsl_nodes.Action):
                if node.children[1].name == node.children[2].name:
                    # Create new action node
                    action_child = dsl_nodes.Action()
                    action_child.name = node.children[1].name
                    
                    # Update parent's children array
                    if node.parent is not None:
                        for i, child in enumerate(node.parent.children):
                            if child == node:
                                node.parent.children[i] = action_child
                                action_child.parent = node.parent
                                break
                    else:
                        # If this is the root node, we need to update the program itself
                        action_child.parent = None
                        return action_child

        # Recursively process children
        for i, child in enumerate(node.children):
            if child is not None:
                result = self._merge_action_nodes(child)
                if result is not None:
                    node.children[i] = result
                    result.parent = node

        return None

    def get_neighbors(self, individual, k = 1, alpha_override: float | None = None) -> list[tuple[dsl_nodes.Program, dsl_nodes.Program]]:
        """Returns k neighbors of a given individual encoded as a program

        Args:
            individual (dsl_nodes.Program): Individual as a program
            k (int, optional): Number of neighbors. Defaults to 1.
            alpha_override (float | None): per-call depth_bias_alpha override
                (iter 15 per-candidate α ensemble). If None, uses
                self.depth_bias_alpha.
        """
        alpha = float(alpha_override) if alpha_override is not None else self.depth_bias_alpha
        neighbors = []
        neighbor_metadata = []
        for _ in range(k):
            accepted = False
            while not accepted:
                mutated_program = copy.deepcopy(individual)
                mutations_meta = []
                for _m in range(self.n_mutations):
                    nodes = mutated_program.get_all_nodes()[1:]
                    if not nodes:
                        break
                    node_to_mutate = self._sample_mutation_node(nodes, alpha=alpha)
                    mutations_meta.append(
                        {
                            "node_type": type(node_to_mutate).__name__,
                            "node_depth": int(node_to_mutate.get_depth()),
                        }
                    )
                    self._mutate_node(node_to_mutate)
                self._merge_action_nodes(mutated_program)
                prog_str = self.dsl.parse_node_to_str(mutated_program)
                accepted = get_max_height(mutated_program) <= self.max_height and get_max_sequence(mutated_program) <= self.max_sequence \
                    and len(prog_str.split(" ")) <= self.max_tokens and len(prog_str.split(" ")) >= self.min_tokens
            neighbors.append((mutated_program, mutated_program))
            neighbor_metadata.append(
                {
                    "mutation_kind": "structural",
                    "mutation_mode": self.mutation_guidance_mode,
                    "mutations": mutations_meta,
                }
            )
        self.last_neighbor_metadata = neighbor_metadata
        return neighbors


    def get_neighbors_with_action(self, individual, k = 1) -> list[tuple[dsl_nodes.Program, dsl_nodes.Program]]:
        """Returns k neighbors of a given individual encoded as a program

        Args:
            individual (dsl_nodes.Program): Individual as a program
            k (int, optional): Number of neighbors. Defaults to 1.

        Returns:
            list[tuple[dsl_nodes.Program, dsl_nodes.Program]]: List of individuals as tuples of
            program (individual) and program (decoding)
        """
        neighbors = []
        neighbor_metadata = []
        for _ in range(k):
            # Easiest way to do a valid mutation is to do a random mutation until we find a valid one
            # This could be changed by restricting the mutation space (_fill_children args in _mutate_node)
            accepted = False
            while not accepted:
                mutated_program = copy.deepcopy(individual)
                action_nodes = [node for node in mutated_program.get_all_nodes() if isinstance(node, dsl_nodes.Action)]
                self._mutate_node_with_action(mutated_program.get_all_nodes(),prob=0.5)
                self._merge_action_nodes(mutated_program)
                prog_str = self.dsl.parse_node_to_str(mutated_program)
                # print(len(prog_str.split(" ")),prog_str,self.max_height,self.max_sequence)
                accepted = get_max_height(mutated_program) <= self.max_height and get_max_sequence(mutated_program) <= self.max_sequence \
                    and len(prog_str.split(" ")) <= self.max_tokens and len(prog_str.split(" ")) >= self.min_tokens
            # print(self.dsl.parse_node_to_str(mutated_program))
            # print(self.max_tokens,len(prog_str.split(" ")), self.dsl.parse_node_to_str(mutated_program))
            # self._merge_action_nodes(mutated_program)
            neighbors.append((mutated_program, mutated_program))
            neighbor_metadata.append(
                {
                    "mutation_kind": "action_resample",
                    "mutation_mode": self.mutation_guidance_mode,
                    "mutations": [
                        {
                            "node_type": "Action",
                            "node_depth": int(node.get_depth()),
                        }
                        for node in action_nodes
                    ],
                }
            )
        self.last_neighbor_metadata = neighbor_metadata
        return neighbors
