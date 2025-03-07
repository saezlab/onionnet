import numpy as np
import pandas as pd
from graph_tool.all import Graph, GraphView, graph_draw, shortest_distance
from graph_tool.topology import label_components, label_out_component
from collections import deque
from typing import Dict, Tuple, List, Any

try:
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

#########################################
# Core Graph and Mappings
#########################################
class OnionNetGraph:
    def __init__(self, directed=True):
        self.graph = Graph(directed=directed)
        # Mapping dictionaries for custom IDs
        self.custom_id_to_vertex_index: Dict[Tuple[int, int], int] = {}
        self.vertex_index_to_custom_id: Dict[int, Tuple[int, int]] = {}
        # Mappings for layer and node IDs
        self.layer_code_to_name: Dict[int, str] = {}
        self.layer_name_to_code: Dict[str, int] = {}
        self.node_id_int_to_str: Dict[int, str] = {}
        self.node_id_str_to_int: Dict[str, int] = {}
        # Mapping for categorical properties
        self.vertex_categorical_mappings: Dict[str, Dict[str, Any]] = {}
        self.edge_categorical_mappings: Dict[str, Dict[str, Any]] = {}

        # Initialize core vertex properties
        self.graph.vp['layer_hash'] = self.graph.new_vertex_property('int64_t')
        self.graph.vp['node_id_hash'] = self.graph.new_vertex_property('int64_t')
    
    def _map_layer(self, layer_name: str) -> int:
        if layer_name in self.layer_name_to_code:
            return self.layer_name_to_code[layer_name]
        else:
            code = len(self.layer_name_to_code)
            self.layer_name_to_code[layer_name] = code
            self.layer_code_to_name[code] = layer_name
            return code

    def _map_node_id(self, node_id_str: str) -> int:
        if node_id_str in self.node_id_str_to_int:
            return self.node_id_str_to_int[node_id_str]
        else:
            code = len(self.node_id_str_to_int)
            self.node_id_str_to_int[node_id_str] = code
            self.node_id_int_to_str[code] = node_id_str
            return code

