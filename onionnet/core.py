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

"""
This module defines the OnionNetGraph class, which serves as the core graph structure for the OnionNet project.
It uses graph-tool for graph representation and provides mapping functions for custom vertex identifiers and layers.
"""

#########################################
# Core Graph and Mappings
#########################################
class OnionNetGraph:
    """
    Core graph structure for the OnionNet project.
    
    This class encapsulates a graph_tool.Graph object and provides methods for mapping custom identifiers,
    layers, and handling categorical properties. It maintains several dictionaries for translating between
    user-defined identifiers and internal representations.
    
    Attributes:
        graph (Graph): The underlying graph_tool.Graph object.
        custom_id_to_vertex_index (Dict[Tuple[int, int], int]): Mapping from custom ID tuple (layer, node_id) to vertex index.
        vertex_index_to_custom_id (Dict[int, Tuple[int, int]]): Reverse mapping from vertex index to custom ID tuple.
        layer_code_to_name (Dict[int, str]): Mapping from layer code to layer name.
        layer_name_to_code (Dict[str, int]): Mapping from layer name to layer code.
        node_id_int_to_str (Dict[int, str]): Mapping from integer node id to its string representation.
        node_id_str_to_int (Dict[str, int]): Mapping from string node id to its integer representation.
        vertex_categorical_mappings (Dict[str, Dict[str, Any]]): Mappings for vertex categorical properties.
        edge_categorical_mappings (Dict[str, Dict[str, Any]]): Mappings for edge categorical properties.
    """
    def __init__(self, directed: bool = True):
        """
        Initialize the OnionNetGraph.
        
        Parameters:
            directed (bool, optional): Determines if the graph is directed. Defaults to True.
        
        The constructor initializes the underlying graph_tool.Graph and sets up dictionaries for custom ID mappings and
        categorical properties. It also creates vertex properties for layer and node identifiers.
        """
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
        
        # Initialize core vertex properties for layer and node identifiers
        self.graph.vp['layer_hash'] = self.graph.new_vertex_property('int64_t')
        self.graph.vp['node_id_hash'] = self.graph.new_vertex_property('int64_t')
    
    def _map_layer(self, layer_name: str) -> int:
        """
        Map a layer name to a unique integer code.
        
        If the layer name already exists, its corresponding code is returned. Otherwise, a new code is assigned
        and the mappings are updated.
        
        Parameters:
            layer_name (str): The name of the layer.
        
        Returns:
            int: The integer code corresponding to the layer.
        """
        if layer_name in self.layer_name_to_code:
            return self.layer_name_to_code[layer_name]
        else:
            code = len(self.layer_name_to_code)
            self.layer_name_to_code[layer_name] = code
            self.layer_code_to_name[code] = layer_name
            return code
    
    def _map_node_id(self, node_id_str: str) -> int:
        """
        Map a node identifier string to a unique integer code.
        
        If the node identifier already exists, its corresponding code is returned. Otherwise, a new code is assigned
        and the mappings are updated.
        
        Parameters:
            node_id_str (str): The node identifier as a string.
        
        Returns:
            int: The integer code corresponding to the node identifier.
        """
        if node_id_str in self.node_id_str_to_int:
            return self.node_id_str_to_int[node_id_str]
        else:
            code = len(self.node_id_str_to_int)
            self.node_id_str_to_int[node_id_str] = code
            self.node_id_int_to_str[code] = node_id_str
            return code
