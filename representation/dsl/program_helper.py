from __future__ import annotations

from typing import Optional, Set

from . import dsl_nodes
from .scheduling import LinearBoolExpr, SchedulingIf, SchedulingITE

def get_linear_param_vector(node_program):
    """
    Extract parameters from all LinearBoolExpr nodes into a single vector.
    Feature names include the full path using dot notation.
    Includes coefficient_0 for constant terms.
    
    Args:
        node_program: Root node of the program tree
        
    Returns:
        param_vector: List of all coefficients concatenated
        feature_paths: List of lists of fully qualified feature paths
    """
    param_vector = []
    feature_paths = []
    param_feature_dict = {}
    
    def traverse(node, path="root"):
        if node is None:
            return
            
        if isinstance(node, LinearBoolExpr):
            # Create fully qualified feature paths including coefficient_0
            full_paths = [f"{path}.LinearBoolExpr.coefficient_0"] + [f"{path}.LinearBoolExpr.{name}" for name in node.feature_names]
            param_vector.extend(node.coefficients)
            feature_paths.append(full_paths)
            for feature_name, coefficient in zip(full_paths, node.coefficients):
                param_feature_dict[feature_name] = coefficient
        elif isinstance(node, dsl_nodes.WeightedAction):
            full_paths = [f"{path}.WeightedAction.{name}" for name in node.action_names]
            param_vector.extend(node.logits)
            feature_paths.append(full_paths)
            for feature_name, logit in zip(full_paths, node.logits):
                param_feature_dict[feature_name] = logit
        if hasattr(node, 'children') and len(node.children) > 0:
            if isinstance(node, dsl_nodes.Program):
                traverse(node.children[0], "root.program")
            elif isinstance(node, SchedulingIf):
                traverse(node.children[0], f"{path}.if.condition")
                traverse(node.children[1], f"{path}.if.body")
            elif isinstance(node, SchedulingITE):
                traverse(node.children[0], f"{path}.ifelse.condition")
                traverse(node.children[1], f"{path}.ifelse.if")
                traverse(node.children[2], f"{path}.ifelse.else")
            elif isinstance(node, dsl_nodes.And):
                traverse(node.children[0], f"{path}.and.left")
                traverse(node.children[1], f"{path}.and.right")
            elif isinstance(node, dsl_nodes.Or):
                traverse(node.children[0], f"{path}.or.left")
                traverse(node.children[1], f"{path}.or.right")
            elif isinstance(node, dsl_nodes.Not):
                traverse(node.children[0], f"{path}.not")
            else:
                raise ValueError(f"Unknown node type: {type(node)}")
    
    traverse(node_program)
    return param_vector, feature_paths, param_feature_dict


def merge_param_dict_parent_into_child(child_root, parent_root):
    """For BO warm-start: use parent's coefficient where the DSL path matches the child."""
    _, _, child_map = get_linear_param_vector(child_root)
    _, _, parent_map = get_linear_param_vector(parent_root)
    merged = dict(child_map)
    for k in merged:
        if k in parent_map:
            merged[k] = parent_map[k]
    return merged


def linear_param_map_from_program(node_program):
    """Flat name -> coefficient map for BO registration / warm-start."""
    _, _, param_map = get_linear_param_vector(node_program)
    return dict(param_map)


def set_linear_param_vector(node_program, param_vector, feature_paths):
    """
    Update all LinearBoolExpr nodes from a parameter vector.
    Expects feature paths including coefficient_0 for constant terms.
    
    Args:
        node_program: Root node of the program tree
        param_vector: List of all coefficients concatenated
        feature_paths: List of lists of fully qualified feature paths
        
    Returns:
        Updated node_program
    """
    vector_index = [0]  # Use list for mutable reference
    node_index = [0]
    
    def traverse(node, path="root"):
        if node is None:
            return
            
        if isinstance(node, LinearBoolExpr):
            # Extract original feature names by removing path prefix
            current_paths = feature_paths[node_index[0]]
            expected_prefix = path + ".LinearBoolExpr."
            
            # Verify coefficient_0 is present and correctly formatted
            if not current_paths or not current_paths[0].endswith('.coefficient_0'):
                raise ValueError(f"First feature path must be coefficient_0, got {current_paths[0]}")
                
            # Verify path matches for coefficient_0
            if not current_paths[0].startswith(expected_prefix):
                raise ValueError(f"Feature path mismatch: expected prefix {expected_prefix} for {current_paths[0]}")
            
            # Extract feature names (excluding coefficient_0)
            feature_names = []
            for full_path in current_paths[1:]:
                if not full_path.startswith(expected_prefix):
                    raise ValueError(f"Feature path mismatch: expected prefix {expected_prefix} for {full_path}")
                feature_name = full_path[len(expected_prefix):]
                if feature_name == 'coefficient_0':
                    raise ValueError(f"coefficient_0 must be the first feature path")
                feature_names.append(feature_name)
            
            num_features = len(feature_names)
            num_coeffs = num_features + 1  # Include constant term
            
            # Extract coefficients for this node
            node.coefficients = param_vector[vector_index[0]:vector_index[0] + num_coeffs]
            node.feature_names = feature_names
            
            vector_index[0] += num_coeffs
            node_index[0] += 1
        elif isinstance(node, dsl_nodes.WeightedAction):
            current_paths = feature_paths[node_index[0]]
            expected_prefix = path + ".WeightedAction."
            action_names = []
            for full_path in current_paths:
                if not full_path.startswith(expected_prefix):
                    raise ValueError(f"Feature path mismatch: expected prefix {expected_prefix} for {full_path}")
                action_names.append(full_path[len(expected_prefix):])
            num_logits = len(action_names)
            node.action_names = action_names
            node.logits = param_vector[vector_index[0]:vector_index[0] + num_logits]
            vector_index[0] += num_logits
            node_index[0] += 1
            
        if hasattr(node, 'children') and len(node.children) > 0:
            if isinstance(node, dsl_nodes.Program):
                traverse(node.children[0], "root.program")
            elif isinstance(node, SchedulingIf):
                traverse(node.children[0], f"{path}.if.condition")
                traverse(node.children[1], f"{path}.if.body")
            elif isinstance(node, SchedulingITE):
                traverse(node.children[0], f"{path}.ifelse.condition")
                traverse(node.children[1], f"{path}.ifelse.if")
                traverse(node.children[2], f"{path}.ifelse.else")
            elif isinstance(node, dsl_nodes.And):
                traverse(node.children[0], f"{path}.and.left")
                traverse(node.children[1], f"{path}.and.right")
            elif isinstance(node, dsl_nodes.Or):
                traverse(node.children[0], f"{path}.or.left")
                traverse(node.children[1], f"{path}.or.right")
            elif isinstance(node, dsl_nodes.Not):
                traverse(node.children[0], f"{path}.not")
            else:
                raise ValueError(f"Unknown node type: {type(node)}")
    
    traverse(node_program)
    return node_program


def find_dsl_path_for_node(root, target, path: str = "root") -> Optional[str]:
    """
    Return the DSL path string for `target` if it appears under `root`, else None.
    Paths match get_linear_param_vector / set_linear_param_vector conventions.
    """
    if root is None:
        return None
    if root is target:
        return path
    if isinstance(root, dsl_nodes.Program):
        if not root.children or root.children[0] is None:
            return None
        return find_dsl_path_for_node(root.children[0], target, "root.program")
    if isinstance(root, SchedulingIf):
        r = find_dsl_path_for_node(root.children[0], target, f"{path}.if.condition")
        if r is not None:
            return r
        return find_dsl_path_for_node(root.children[1], target, f"{path}.if.body")
    if isinstance(root, SchedulingITE):
        r = find_dsl_path_for_node(root.children[0], target, f"{path}.ifelse.condition")
        if r is not None:
            return r
        r = find_dsl_path_for_node(root.children[1], target, f"{path}.ifelse.if")
        if r is not None:
            return r
        return find_dsl_path_for_node(root.children[2], target, f"{path}.ifelse.else")
    if isinstance(root, dsl_nodes.And):
        r = find_dsl_path_for_node(root.children[0], target, f"{path}.and.left")
        if r is not None:
            return r
        return find_dsl_path_for_node(root.children[1], target, f"{path}.and.right")
    if isinstance(root, dsl_nodes.Or):
        r = find_dsl_path_for_node(root.children[0], target, f"{path}.or.left")
        if r is not None:
            return r
        return find_dsl_path_for_node(root.children[1], target, f"{path}.or.right")
    if isinstance(root, dsl_nodes.Not):
        return find_dsl_path_for_node(root.children[0], target, f"{path}.not")
    return None


def collect_linear_bool_param_keys(node, path: str, out: Set[str]) -> None:
    """Accumulate fully-qualified LinearBoolExpr parameter keys under `node` at DSL `path`."""
    if node is None:
        return
    if isinstance(node, LinearBoolExpr):
        keys = [f"{path}.LinearBoolExpr.coefficient_0"] + [
            f"{path}.LinearBoolExpr.{name}" for name in node.feature_names
        ]
        out.update(keys)
        return
    if not hasattr(node, "children") or not node.children:
        return
    if isinstance(node, dsl_nodes.Program):
        collect_linear_bool_param_keys(node.children[0], "root.program", out)
    elif isinstance(node, SchedulingIf):
        collect_linear_bool_param_keys(node.children[0], f"{path}.if.condition", out)
        collect_linear_bool_param_keys(node.children[1], f"{path}.if.body", out)
    elif isinstance(node, SchedulingITE):
        collect_linear_bool_param_keys(node.children[0], f"{path}.ifelse.condition", out)
        collect_linear_bool_param_keys(node.children[1], f"{path}.ifelse.if", out)
        collect_linear_bool_param_keys(node.children[2], f"{path}.ifelse.else", out)
    elif isinstance(node, dsl_nodes.And):
        collect_linear_bool_param_keys(node.children[0], f"{path}.and.left", out)
        collect_linear_bool_param_keys(node.children[1], f"{path}.and.right", out)
    elif isinstance(node, dsl_nodes.Or):
        collect_linear_bool_param_keys(node.children[0], f"{path}.or.left", out)
        collect_linear_bool_param_keys(node.children[1], f"{path}.or.right", out)
    elif isinstance(node, dsl_nodes.Not):
        collect_linear_bool_param_keys(node.children[0], f"{path}.not", out)
