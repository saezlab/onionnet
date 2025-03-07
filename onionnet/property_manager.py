from .core import OnionNetGraph
from .utils import infer_property_type, map_categorical_property
from typing import List, Any, Dict



#########################################
# Property Manager: Access and Conversion
#########################################
class OnionNetPropertyManager:
    def __init__(self, core: OnionNetGraph):
        self.core = core

    def get_vertex_by_encoding_tuple(self, layer_code: int, node_id_int: int):
        idx = self.core.custom_id_to_vertex_index.get((layer_code, node_id_int))
        return self.core.graph.vertex(idx) if idx is not None else None

    def get_vertex_by_name_tuple(self, layer_name: str, node_id_str: str):
        layer_code = self.core.layer_name_to_code.get(layer_name)
        node_id_int = self.core.node_id_str_to_int.get(node_id_str)
        if layer_code is None or node_id_int is None:
            raise KeyError("Layer or node ID not found.")
        return self.get_vertex_by_encoding_tuple(layer_code, node_id_int)

    def get_vertex_property(self, layer_code: int, node_id_int: int, prop_name: str):
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is not None and prop_name in self.core.graph.vp:
            return self.core.graph.vp[prop_name][v]
        return None

    def set_vertex_property(self, layer_code: int, node_id_int: int, prop_name: str, value: Any):
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is not None:
            if prop_name not in self.core.graph.vp:
                typ = infer_property_type(value)
                self.core.graph.vp[prop_name] = self.core.graph.new_vertex_property(typ)
            self.core.graph.vp[prop_name][v] = value
        else:
            print(f"Vertex ({layer_code}, {node_id_int}) not found.")

    def view_node_properties(self, layer_code: int, node_id_int: int) -> Dict[str, Any]:
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is None:
            print("Vertex not found.")
            return {}
        props = {}
        for p in self.core.graph.vp.keys():
            val = self.core.graph.vp[p][v]
            if p in self.core.vertex_categorical_mappings:
                mapping = self.core.vertex_categorical_mappings[p]['int_to_str']
                val = mapping.get(val, f"Unknown ({val})")
            props[p] = val
        props['decoded_layer'] = self.core.layer_code_to_name.get(layer_code, f"Unknown ({layer_code})")
        props['decoded_node_id'] = self.core.node_id_int_to_str.get(node_id_int, f"Unknown ({node_id_int})")
        return props

    def view_node_properties_by_names(self, layer_name: str, node_id_str: str, verbose: bool = False) -> Dict[str, Any]:
        v = self.get_vertex_by_name_tuple(layer_name, node_id_str)
        props = self.view_node_properties(self.core.layer_name_to_code[layer_name],
                                          self.core.node_id_str_to_int[node_id_str])
        if verbose:
            print(f"Properties for ({layer_name}, {node_id_str}):")
            for k, val in props.items():
                print(f"  {k}: {val}")
        return props

    def create_node_label_property(self, prop_name: str = 'node_label') -> None:
        if prop_name in self.core.graph.vp:
            print(f"Property '{prop_name}' already exists.")
            return
        label_prop = self.core.graph.new_vertex_property('string')
        for v in self.core.graph.vertices():
            layer = self.core.layer_code_to_name.get(self.core.graph.vp['layer_hash'][v], "Unknown")
            nid = self.core.node_id_int_to_str.get(self.core.graph.vp['node_id_hash'][v], "Unknown")
            label_prop[v] = f"{layer}:{nid}"
        self.core.graph.vp[prop_name] = label_prop
        print(f"Vertex property '{prop_name}' created successfully.")

    def decode_property_labels(
        self, 
        encoded_prop_type: str,  # 'v' for vertex, 'e' for edge
        encoded_prop_name: str, 
        new_prop_name: str = None, # defaults to f"{encoded_prop_name}_decoded"
        mapping_dict: Dict[int, str] = None,  # defaults to self.core.vertex_categorical_mappings[encoded_prop_name]['int_to_str']
        default_unmapped_label: str = 'Unknown'
    ) -> None:
        """
        Creates a new property by mapping encoded integers to human-readable strings.

        Parameters:
            encoded_prop_type (str): Type of the property ('v' for vertex, 'e' for edge).
            encoded_prop_name (str): Name of the existing encoded property.
            new_prop_name (str, optional): Name of the new human-readable property to create. Defaults to f"{encoded_prop_name}_decoded"
            mapping_dict (Dict[int, str], optional): Dictionary mapping encoded integers to strings. Defaults to self.core.vertex_categorical_mappings[encoded_prop_name]['int_to_str'] or self.core.edge_categorical_mappings[encoded_prop_name]['int_to_str'], depending on whether encoded_prop_type is 'v' or 'e'.
            default_unmapped_label (str, optional): Default label for unmapped integers. Defaults to 'Unknown'.
        
        Raises:
            ValueError: If encoded_prop_type is not 'v' or 'e'.
            KeyError: If the specified property does not exist in the graph.

        Example usage:
            onion.decode_property_labels(
                encoded_prop_type='v', 
                encoded_prop_name='layer_hash', 
                new_prop_name='layer_name',
                mapping_dict=onion.layer_code_to_name
            )
        """
        if encoded_prop_type not in ['v', 'e']:
            raise ValueError("encoded_prop_type must be 'v' for vertex or 'e' for edge.")
        
        if new_prop_name is None:
            new_prop_name = f"{encoded_prop_name}_decoded"

        if mapping_dict is None:
            if encoded_prop_type=='v':
                mapping_dict = self.core.vertex_categorical_mappings[encoded_prop_name]['int_to_str']
            else: # must be encoded_prop_type=='e':
                mapping_dict = self.core.edge_categorical_mappings[encoded_prop_name]['int_to_str']
        
        if encoded_prop_type == 'v':
            if encoded_prop_name not in self.core.graph.vp:
                raise KeyError(f"Vertex property '{encoded_prop_name}' does not exist.")
            prop = self.core.graph.vp[encoded_prop_name]
            items = list(self.core.graph.vertices())
        else:
            if encoded_prop_name not in self.graph.ep:
                raise KeyError(f"Edge property '{encoded_prop_name}' does not exist.")
            prop = self.core.graph.ep[encoded_prop_name]
            items = list(self.graph.edges())
        
        # Create a new string property to hold the human-readable labels
        human_readable_prop = self.core.graph.new_property(encoded_prop_type, 'string')
        
        # Generate labels in one pass over the items
        labels = [mapping_dict.get(int(prop[item]), default_unmapped_label) for item in items]
        
        # Assign the labels to the new property
        for item, label in zip(items, labels):
            human_readable_prop[item] = label
        
        # Add the new property to the appropriate property map
        if encoded_prop_type == 'v':
            self.core.graph.vp[new_prop_name] = human_readable_prop
        else:
            self.core.graph.ep[new_prop_name] = human_readable_prop
        
        print(f"{encoded_prop_type.upper()} property '{new_prop_name}' created successfully.")