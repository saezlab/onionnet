from typing import Dict, Tuple
from graph_tool.all import GraphView
from .core import OnionNetGraph
from .builder import OnionNetBuilder
from .searcher import OnionNetSearcher
from .property_manager import OnionNetPropertyManager

"""
This module provides the OnionNet class, a high-level interface for managing and interacting with
an OnionNet graph structure. It integrates building, searching, and property management functionalities,
allowing users to grow the graph, perform searches, view components, and manage vertex properties.
"""


class OnionNet:
    """
    High-level interface for the OnionNet graph.
    
    This class encapsulates the core graph and provides APIs to build, search, and manage properties
    of the graph. It uses an underlying OnionNetGraph along with builder, searcher, and property manager
    components to perform operations on the graph.
    
    Attributes:
        core (OnionNetGraph): The core graph object.
        builder (OnionNetBuilder): Component for adding nodes and edges to the graph.
        searcher (OnionNetSearcher): Component for querying and viewing graph subsets.
        prop_manager (OnionNetPropertyManager): Component for managing vertex properties.
        _node_map (Dict[Tuple[str, str], int]): Internal cache mapping (layer, node) to vertex index.
    """
    def __init__(self, directed: bool = True):
        """
        Initialize the OnionNet instance.
        
        Parameters:
            directed (bool, optional): If True, the underlying graph will be directed. Defaults to True.
        """
        self.core = OnionNetGraph(directed)
        self.builder = OnionNetBuilder(self.core)
        self.searcher = OnionNetSearcher(self.core)
        self.prop_manager = OnionNetPropertyManager(self.core)
        self._node_map = None

    # Build-related API
    def grow_onion(self, *args, **kwargs) -> None:
        """
        Grow the OnionNet graph by adding nodes and edges.
        
        Delegates to the builder's grow_onion method. Resets the internal node_map cache after growing the graph.
        
        Parameters:
            *args: Positional arguments forwarded to the builder.
            **kwargs: Keyword arguments forwarded to the builder.
        """
        self.builder.grow_onion(*args, **kwargs)
        self._node_map = None  # reset cache if graph changes

    # Search-related API
    def search(self, *args, **kwargs) -> GraphView:
        """
        Perform a search on the OnionNet graph.
        
        Delegates to the searcher's search method.
        
        Returns:
            GraphView: A view of the graph based on the search criteria.
        """
        return self.searcher.search(*args, **kwargs)

    def view_layers(self, *args, **kwargs) -> GraphView:
        """
        View different layers of the OnionNet graph.
        
        Delegates to the searcher's view_layers method.
        
        Returns:
            GraphView: A view of the graph filtered by layers.
        """
        return self.searcher.view_layers(*args, **kwargs)

    def view_components(self, *args, **kwargs) -> GraphView:
        """
        View connected components of the OnionNet graph.
        
        Delegates to the searcher's view_components method.
        
        Returns:
            GraphView: A view of the graph filtered by connected components.
        """
        return self.searcher.view_components(*args, **kwargs)

    def filter_view_by_property(self, *args, **kwargs) -> GraphView:
        """
        Filter the graph view based on vertex or edge properties.
        
        Delegates to the searcher's filter_view_by_property method.
        
        Returns:
            GraphView: A view of the graph filtered by the specified property criteria.
        """
        return self.searcher.filter_view_by_property(*args, **kwargs)
    
    def compose_filters(self, *args, **kwargs) -> GraphView:
        """
        Compose multiple filters to obtain a refined graph view.
        
        Delegates to the searcher's compose_filters method.
        
        Returns:
            GraphView: A view of the graph after applying composed filters.
        """
        return self.searcher.compose_filters(*args, **kwargs)
    
    def create_bipartite_gv(self, *args, **kwargs) -> GraphView:
        """
        Create a bipartite graph view from the OnionNet graph.
        
        Delegates to the searcher's create_bipartite_gv method.
        
        Returns:
            GraphView: A bipartite view of the graph.
        """
        return self.searcher.create_bipartite_gv(*args, **kwargs)

    # Property-related API
    def get_vertex_by_encoding_tuple(self, *args, **kwargs):
        """
        Retrieve a vertex based on its encoding tuple.
        
        Delegates to the property manager's get_vertex_by_encoding_tuple method.
        """
        return self.prop_manager.get_vertex_by_encoding_tuple(*args, **kwargs)

    def get_vertex_by_name_tuple(self, *args, **kwargs):
        """
        Retrieve a vertex based on its name tuple.
        
        Delegates to the property manager's get_vertex_by_name_tuple method.
        """
        return self.prop_manager.get_vertex_by_name_tuple(*args, **kwargs)

    def get_vertex_property(self, *args, **kwargs):
        """
        Get a property value of a vertex.
        
        Delegates to the property manager's get_vertex_property method.
        """
        return self.prop_manager.get_vertex_property(*args, **kwargs)

    def set_vertex_property(self, *args, **kwargs) -> None:
        """
        Set a property value for a vertex.
        
        Delegates to the property manager's set_vertex_property method.
        """
        self.prop_manager.set_vertex_property(*args, **kwargs)

    def view_node_properties(self, *args, **kwargs):
        """
        View all properties of nodes in the graph.
        
        Delegates to the property manager's view_node_properties method.
        """
        return self.prop_manager.view_node_properties(*args, **kwargs)

    def view_node_properties_by_names(self, *args, **kwargs):
        """
        View node properties filtered by specified names.
        
        Delegates to the property manager's view_node_properties_by_names method.
        """
        return self.prop_manager.view_node_properties_by_names(*args, **kwargs)

    def create_node_label_property(self, *args, **kwargs) -> None:
        """
        Create a node label property for the graph.
        
        Delegates to the property manager's create_node_label_property method.
        """
        self.prop_manager.create_node_label_property(*args, **kwargs)

    @property
    def node_map(self) -> Dict[Tuple[str, str], int]:
        """
        Get a mapping from (layer, node) to vertex index.
        
        This property builds and returns a dictionary that maps a tuple of (layer, node name)
        to the corresponding vertex index in the graph. The mapping is cached internally for efficiency.
        
        Returns:
            Dict[Tuple[str, str], int]: A dictionary mapping (layer, node) to vertex index.
        """
        if self._node_map is None:
            self._node_map = {}
            for (layer_code, node_id_int), idx in self.core.custom_id_to_vertex_index.items():
                layer = self.core.layer_code_to_name.get(layer_code, f"Unknown ({layer_code})")
                node = self.core.node_id_int_to_str.get(node_id_int, f"Unknown ({node_id_int})")
                self._node_map[(layer, node)] = idx
        return self._node_map
    
    @property
    def g(self):
        """
        Shortcut to access the underlying graph from the OnionNet instance.
        
        Returns:
            Graph: The core graph contained in the OnionNetGraph.
        """
        return self.core.graph