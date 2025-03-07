from typing import Dict, Tuple
from graph_tool.all import GraphView
from .core import OnionNetGraph
from .builder import OnionNetBuilder
from .searcher import OnionNetSearcher
from .property_manager import OnionNetPropertyManager

class OnionNet:
    def __init__(self, directed=True):
        self.core = OnionNetGraph(directed)
        self.builder = OnionNetBuilder(self.core)
        self.searcher = OnionNetSearcher(self.core)
        self.prop_manager = OnionNetPropertyManager(self.core)
        self._node_map = None

    # Build-related API
    def grow_onion(self, *args, **kwargs) -> None:
        self.builder.grow_onion(*args, **kwargs)
        self._node_map = None  # reset cache if graph changes

    # Search-related API
    def search(self, *args, **kwargs) -> GraphView:
        return self.searcher.search(*args, **kwargs)

    def view_layers(self, *args, **kwargs) -> GraphView:
        return self.searcher.view_layers(*args, **kwargs)

    def view_components(self, *args, **kwargs) -> GraphView:
        return self.searcher.view_components(*args, **kwargs)

    def filter_view_by_property(self, *args, **kwargs) -> GraphView:
        return self.searcher.filter_view_by_property(*args, **kwargs)

    # Property-related API
    def get_vertex_by_encoding_tuple(self, *args, **kwargs):
        return self.prop_manager.get_vertex_by_encoding_tuple(*args, **kwargs)

    def get_vertex_by_name_tuple(self, *args, **kwargs):
        return self.prop_manager.get_vertex_by_name_tuple(*args, **kwargs)

    def get_vertex_property(self, *args, **kwargs):
        return self.prop_manager.get_vertex_property(*args, **kwargs)

    def set_vertex_property(self, *args, **kwargs) -> None:
        self.prop_manager.set_vertex_property(*args, **kwargs)

    def view_node_properties(self, *args, **kwargs):
        return self.prop_manager.view_node_properties(*args, **kwargs)

    def view_node_properties_by_names(self, *args, **kwargs):
        return self.prop_manager.view_node_properties_by_names(*args, **kwargs)

    def create_node_label_property(self, *args, **kwargs) -> None:
        self.prop_manager.create_node_label_property(*args, **kwargs)

    @property
    def node_map(self) -> Dict[Tuple[str, str], int]:
        if self._node_map is None:
            self._node_map = {}
            for (layer_code, node_id_int), idx in self.core.custom_id_to_vertex_index.items():
                layer = self.core.layer_code_to_name.get(layer_code, f"Unknown ({layer_code})")
                node = self.core.node_id_int_to_str.get(node_id_int, f"Unknown ({node_id_int})")
                self._node_map[(layer, node)] = idx
        return self._node_map