"""Core graph structures and mappings for OnionNet.

This module defines the ``OnionNetGraph`` class, the central graph
container used by OnionNet. It manages mappings between human-readable
layer/node identifiers and their integer encodings and provides helpers
to allocate core vertex properties.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from graph_tool.all import Graph

# Detect availability of IPython without importing display to avoid unused imports
try:
    IPYTHON_AVAILABLE = importlib.util.find_spec("IPython") is not None
except Exception:
    IPYTHON_AVAILABLE = False

# (former duplicate module description removed to keep a single module docstring)


#########################################
# Core Graph and Mappings
#########################################
class OnionNetGraph:
    """
    Core graph structure for the OnionNet project.

    This class encapsulates a graph_tool.Graph object and provides methods for mapping custom identifiers,
    layers, and handling categorical properties. It maintains several dictionaries for translating between
    user-defined identifiers and internal representations.

    Attributes
    ----------
    graph : graph_tool.Graph
        The underlying ``graph_tool.Graph`` object.
    custom_id_to_vertex_index : dict of (tuple of (int, int), int)
        Mapping from custom ID tuple ``(layer, node_id)`` to vertex index.
    vertex_index_to_custom_id : dict of (int, tuple of (int, int))
        Reverse mapping from vertex index to custom ID tuple.
    layer_code_to_name : dict of (int, str)
        Mapping from layer code to layer name.
    layer_name_to_code : dict of (str, int)
        Mapping from layer name to layer code.
    node_id_int_to_str : dict of (int, str)
        Mapping from integer node ID to its string representation.
    node_id_str_to_int : dict of (str, int)
        Mapping from string node ID to its integer representation.
    vertex_categorical_mappings : dict of (str, dict of (str, Any))
        Mappings for vertex categorical properties.
    edge_categorical_mappings : dict of (str, dict of (str, Any))
        Mappings for edge categorical properties.
    """

    def __init__(self, directed: bool = True):
        """
        Initialize the OnionNetGraph.

        Parameters
        ----------
        directed : bool, optional
            Determines if the graph is directed. Defaults to True.

        Notes
        -----
        Initializes the underlying ``graph_tool.Graph`` and sets up dictionaries for
        custom ID mappings and categorical properties. Also creates vertex properties
        for layer and node identifiers.
        """
        self.graph = Graph(directed=directed)

        # Mapping dictionaries for custom IDs
        self.custom_id_to_vertex_index: dict[tuple[int, int], int] = {}
        self.vertex_index_to_custom_id: dict[int, tuple[int, int]] = {}

        # Mappings for layer and node IDs
        self.layer_code_to_name: dict[int, str] = {}
        self.layer_name_to_code: dict[str, int] = {}
        self.node_id_int_to_str: dict[int, str] = {}
        self.node_id_str_to_int: dict[str, int] = {}

        # Mapping for categorical properties
        self.vertex_categorical_mappings: dict[str, dict[str, Any]] = {}
        self.edge_categorical_mappings: dict[str, dict[str, Any]] = {}

        # Initialize core vertex properties for layer and node identifiers
        self.graph.vp["layer_hash"] = self.graph.new_vertex_property("int64_t")
        self.graph.vp["node_id_hash"] = self.graph.new_vertex_property("int64_t")

    def _map_layer(self, layer_name: str) -> int:
        """
        Map a layer name to a unique integer code.

        If the layer name already exists, its corresponding code is returned. Otherwise, a new code is assigned
        and the mappings are updated.

        Parameters
        ----------
        layer_name : str
            The name of the layer.

        Returns
        -------
        int
            The integer code corresponding to the layer.
        """
        # Treat layer names as-is (no whitespace normalization)
        key = str(layer_name)
        if key in self.layer_name_to_code:
            return self.layer_name_to_code[key]
        code = len(self.layer_name_to_code)
        self.layer_name_to_code[key] = code
        self.layer_code_to_name[code] = key
        return code

    def _map_node_id(self, node_id_str: str) -> int:
        """
        Map a node identifier string to a unique integer code.

        If the node identifier already exists, its corresponding code is returned. Otherwise, a new code is assigned
        and the mappings are updated.

        Parameters
        ----------
        node_id_str : str
            The node identifier as a string.

        Returns
        -------
        int
            The integer code corresponding to the node identifier.
        """
        if node_id_str in self.node_id_str_to_int:
            return self.node_id_str_to_int[node_id_str]
        code = len(self.node_id_str_to_int)
        self.node_id_str_to_int[node_id_str] = code
        self.node_id_int_to_str[code] = node_id_str
        return code
