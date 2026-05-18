import sys
sys.path.append('.')

from representation.dsl.program_helper import get_linear_param_vector, set_linear_param_vector
from representation.dsl.scheduling import esDSL

class JJSProgram:
    def __init__(self, dsl, node_program=None):
        self.dsl = dsl
        self._node_program = None
        # Optional: dead-branch detector labels (path id -> "P-dead" | "D-dead") for LS tooling.
        self.dead_branch_status: dict = {}

        #parameter for linearBoolExpr
        self._parameters = []
        #feature list for linearBoolExpr
        self._feature_lists = []
        if node_program is not None:
            self.init_from_node(node_program)


    def init_from_node(self, node_program):
        if self._node_program is not None:
            raise ValueError("Program already initialized")
        self._node_program = node_program
        self._parameters, self._feature_lists, self._param_feature_dict = get_linear_param_vector(self._node_program)

    def refresh_from_current_node(self):
        if self._node_program is None:
            raise ValueError("Program not initialized")
        self._parameters, self._feature_lists, self._param_feature_dict = get_linear_param_vector(
            self._node_program
        )
        return self._parameters, self._feature_lists

    def get_linear_param_vector_from_program(self, program):
        return get_linear_param_vector(program)

    def get_linear_param_vector(self):
        return self._parameters, self._feature_lists
    
    def set_linear_param_vector_params(self, params, feature_lists):
        # make sure the params are in the same order as the feature_lists
        param_vector = []
        flattened_feature_lists = [feature for feature_list in feature_lists for feature in feature_list]
        for feature in flattened_feature_lists:
            param_vector.append(params[feature])
        self._parameters = param_vector
        self._feature_lists = feature_lists
        self._node_program = set_linear_param_vector(self._node_program, param_vector, feature_lists)

    def get_node_program(self):
        return self._node_program
