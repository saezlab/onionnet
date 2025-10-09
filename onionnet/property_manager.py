from .core import OnionNetGraph
from .utils import infer_property_type, map_categorical_property
from typing import List, Any, Dict
import numpy as np
import pandas as pd

from graph_tool.all import Graph, GraphView

"""
This module provides the OnionNetPropertyManager class, which handles property access, conversion, and management for vertices (and edges) in an OnionNetGraph.
It includes methods to get and set vertex properties, view properties, create human-readable labels, and decode encoded property values.
"""

#########################################
# Property Manager: Access and Conversion
#########################################
class OnionNetPropertyManager:
    def __init__(self, core: OnionNetGraph):
        """
        Initialize the OnionNetPropertyManager with a core OnionNetGraph instance.
        
        Parameters:
            core (OnionNetGraph): The core graph object that holds the graph and its mappings.
        """
        self.core = core

    def get_vertex_by_encoding_tuple(self, layer_code: int, node_id_int: int):
        """
        Retrieve a vertex from the graph using its encoded identifier tuple.
        
        Parameters:
            layer_code (int): The integer code representing the layer.
            node_id_int (int): The integer code representing the node identifier.
        
        Returns:
            Vertex: The corresponding vertex object if found; otherwise, None.
        """
        idx = self.core.custom_id_to_vertex_index.get((layer_code, node_id_int))
        return self.core.graph.vertex(idx) if idx is not None else None

    def get_vertex_by_name_tuple(self, layer_name: str, node_id_str: str):
        """
        Retrieve a vertex from the graph using its human-readable name tuple.
        
        Parameters:
            layer_name (str): The name of the layer.
            node_id_str (str): The string representation of the node identifier.
        
        Returns:
            Vertex: The corresponding vertex object if found.
        
        Raises:
            KeyError: If the layer or node ID is not found in the graph's mappings.
        """
        layer_code = self.core.layer_name_to_code.get(layer_name)
        node_id_int = self.core.node_id_str_to_int.get(node_id_str)
        if layer_code is None or node_id_int is None:
            raise KeyError("Layer or node ID not found.")
        return self.get_vertex_by_encoding_tuple(layer_code, node_id_int)

    def get_vertex_property(self, layer_code: int, node_id_int: int, prop_name: str):
        """
        Get the value of a specified property for a given vertex.
        
        Parameters:
            layer_code (int): The integer code representing the layer.
            node_id_int (int): The integer code representing the node identifier.
            prop_name (str): The name of the property to retrieve.
        
        Returns:
            Any: The value of the property if it exists; otherwise, None.
        """
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is not None and prop_name in self.core.graph.vp:
            return self.core.graph.vp[prop_name][v]
        return None

    def set_vertex_property(self, layer_code: int, node_id_int: int, prop_name: str, value: Any):
        """
        Set the value of a specified property for a given vertex.
        
        If the property does not already exist, it is created using the inferred type of the provided value.
        
        Parameters:
            layer_code (int): The integer code representing the layer.
            node_id_int (int): The integer code representing the node identifier.
            prop_name (str): The name of the property to set.
            value (Any): The value to assign to the property.
        
        Side Effects:
            Updates the property map of the graph with the new value.
        
        Notes:
            If the vertex is not found, a message is printed.
        """
        v = self.get_vertex_by_encoding_tuple(layer_code, node_id_int)
        if v is not None:
            if prop_name not in self.core.graph.vp:
                typ = infer_property_type(value)
                self.core.graph.vp[prop_name] = self.core.graph.new_vertex_property(typ)
            self.core.graph.vp[prop_name][v] = value
        else:
            print(f"Vertex ({layer_code}, {node_id_int}) not found.")

    def view_node_properties(self, layer_code: int, node_id_int: int) -> Dict[str, Any]:
        """
        View all properties for a specified vertex.
        
        Parameters:
            layer_code (int): The integer code representing the layer.
            node_id_int (int): The integer code representing the node identifier.
        
        Returns:
            Dict[str, Any]: A dictionary of property names and their corresponding values for the vertex.
                          Includes decoded layer and node identifiers.
        
        Side Effects:
            If the vertex is not found, prints an error message and returns an empty dictionary.
        """
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
        """
        View properties for a vertex using its human-readable layer and node identifier.
        
        Parameters:
            layer_name (str): The name of the layer.
            node_id_str (str): The string representation of the node identifier.
            verbose (bool, optional): If True, prints the properties. Defaults to False.
        
        Returns:
            Dict[str, Any]: A dictionary of property names and their corresponding values.
        
        Side Effects:
            If verbose is True, prints the properties to the console.
        """
        v = self.get_vertex_by_name_tuple(layer_name, node_id_str)
        props = self.view_node_properties(self.core.layer_name_to_code[layer_name],
                                          self.core.node_id_str_to_int[node_id_str])
        if verbose:
            print(f"Properties for ({layer_name}, {node_id_str}):")
            for k, val in props.items():
                print(f"  {k}: {val}")
        return props

    def create_node_label_property(self, prop_name: str = 'node_label') -> None:
        """
        Create a new vertex property that serves as a label, combining layer and node identifiers.
        
        The label is formatted as 'layer:node_id'. If the property already exists, a message is printed.
        
        Parameters:
            prop_name (str, optional): The name of the new label property. Defaults to 'node_label'.
        
        Side Effects:
            Adds a new vertex property to the graph and prints a confirmation message.
        """
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
        new_prop_name: str = None,  # Defaults to f"{encoded_prop_name}_decoded"
        mapping_dict: Dict[int, str] = None,  # Defaults based on core's mappings
        default_label: str = 'Unknown',
        g: Graph = None,
    ) -> None:
        """
        Create a new property by mapping encoded integer values to human-readable strings.
        
        Uses NumPy vectorized operations to apply the mapping across the property values.
        
        Parameters:
            encoded_prop_type (str): 'v' for vertex or 'e' for edge.
            encoded_prop_name (str): Name of the existing encoded property.
            new_prop_name (str, optional): Name of the new property. Defaults to f"{encoded_prop_name}_decoded".
            mapping_dict (Dict[int, str], optional): Dictionary mapping integer codes to strings.
                If not provided, defaults to the core's categorical mapping for the property.
            default_label (str): The label to use if a value is not found in the mapping dictionary.
        
        Side Effects:
            Adds a new property to the graph with human-readable labels and prints a confirmation message.
        
        Raises:
            ValueError: If the encoded_prop_type is not 'v' or 'e', or if property conversion fails.
            KeyError: If the specified encoded property does not exist.
        """
        if encoded_prop_type not in ('v', 'e'):
            raise ValueError("encoded_prop_type must be 'v' (vertex) or 'e' (edge).")

        if new_prop_name is None:
            new_prop_name = f"{encoded_prop_name}_decoded"

        # Operate on the provided graph/view (or the core graph by default)
        active = g if g is not None else self.core.graph
        is_view = isinstance(active, GraphView)

        # ---- dimension mismatch guard with nuance ----
        if encoded_prop_type == 'v':
            # If it's not on vertices but *is* known on edges → dimension mismatch (ValueError)
            if encoded_prop_name not in active.vp and encoded_prop_name in self.core.edge_categorical_mappings:
                raise ValueError(
                    f"'{encoded_prop_name}' looks like an EDGE property; cannot decode it as a VERTEX property."
                )
        else:  # 'e'
            if encoded_prop_name not in active.ep and encoded_prop_name in self.core.vertex_categorical_mappings:
                raise ValueError(
                    f"'{encoded_prop_name}' looks like a VERTEX property; cannot decode it as an EDGE property."
                )

        # Ensure the encoded property exists on the requested dimension
        if encoded_prop_type == 'v':
            if encoded_prop_name not in active.vp:
                raise KeyError(f"Vertex property '{encoded_prop_name}' does not exist.")
        else:
            if encoded_prop_name not in active.ep:
                raise KeyError(f"Edge property '{encoded_prop_name}' does not exist.")

        # Resolve mapping dict if not provided.
        # Special-case edge-side layer props to use the global layer map.
        if mapping_dict is None:
            if encoded_prop_type == 'v':
                if encoded_prop_name not in self.core.vertex_categorical_mappings:
                    raise ValueError(
                        f"Vertex property '{encoded_prop_name}' is not registered as categorical; "
                        f"provide mapping_dict to decode."
                    )
                mapping_dict = self.core.vertex_categorical_mappings[encoded_prop_name]['int_to_str']
            else:
                if encoded_prop_name in ('source_layer', 'target_layer'):
                    mapping_dict = self.core.layer_code_to_name
                else:
                    if encoded_prop_name not in self.core.edge_categorical_mappings:
                        raise ValueError(
                            f"Edge property '{encoded_prop_name}' is not registered as categorical; "
                            f"provide mapping_dict to decode."
                        )
                    mapping_dict = self.core.edge_categorical_mappings[encoded_prop_name]['int_to_str']

        # Helper: vectorized lookup → array of labels (object dtype)
        def vectorized_labels(int_array: np.ndarray) -> np.ndarray:
            # simple + robust: vectorize mapping with default fallback
            return np.vectorize(lambda x: mapping_dict.get(int(x), default_label), otypes=[object])(int_array)

        if encoded_prop_type == 'e':
            # ---- EDGE BRANCH ----
            if encoded_prop_name not in active.ep:
                raise KeyError(f"Edge property '{encoded_prop_name}' does not exist.")

            decoded = active.new_edge_property('string')

            if is_view:
                # Safe on any graph-tool version; no base unwrap needed
                enc_map = active.ep[encoded_prop_name]
                for e in active.edges():
                    code = int(enc_map[e])
                    decoded[e] = mapping_dict.get(code, default_label)
            else:
                # Keep the fast vectorized path on full graphs
                enc = active.ep[encoded_prop_name].a.astype(int)
                table = np.empty(max(int(enc.max()) + 1, 1), dtype=object)
                table[:] = default_label
                for k, v in mapping_dict.items():
                    if isinstance(k, (int, np.integer)) and 0 <= int(k) < table.size:
                        table[int(k)] = v
                labels_all = table[enc]
                for e in active.edges():
                    eid = int(active.edge_index[e])
                    decoded[e] = labels_all[eid]

            active.ep[new_prop_name] = decoded
            print(f"E property '{new_prop_name}' created successfully.")

        else:
            # ---- VERTEX BRANCH ----
            if encoded_prop_name not in active.vp:
                raise KeyError(f"Vertex property '{encoded_prop_name}' does not exist.")
            decoded = active.new_vertex_property('string')

            if is_view:
                enc_map = active.vp[encoded_prop_name]
                for v in active.vertices():
                    code = int(enc_map[v])
                    decoded[v] = mapping_dict.get(code, default_label)
            else:
                enc = active.vp[encoded_prop_name].a.astype(int)
                labels_all = np.vectorize(lambda x: mapping_dict.get(x, default_label), otypes=[object])(enc)
                for v in active.vertices():
                    decoded[v] = labels_all[int(v)]

            active.vp[new_prop_name] = decoded
            print(f"V property '{new_prop_name}' created successfully.")


    def decode_property_labels_bulk(
        self,
        df: pd.DataFrame,
        encoded_prop_type: str = 'v',
        g: Graph = None,
    ) -> None:
        """
        Bulk decode categorical properties based on DataFrame columns.

        Iterates over each column in the provided DataFrame and applies decode_property_labels
        to object-type columns, creating a decoded property for each via mapping integer codes
        to human-readable strings.

        Parameters:
            df (pd.DataFrame): DataFrame containing columns corresponding to encoded properties.
            encoded_prop_type (str, optional): 'v' for vertex properties or 'e' for edge properties.
                Defaults to 'v'.

        Side Effects:
            For each object-type column in df, adds a new decoded property to the graph by calling
            decode_property_labels. Prints a message for non-object-type columns indicating no decoding
            was performed.

        Raises:
            KeyError: If decode_property_labels raises a KeyError for a missing property.
        """
        def clean_colname(colname: str) -> str:
            return (
                colname.replace(' ', '_')
                       .replace('(', '')
                       .replace(')', '')
                       .replace('/', '_')
                       .replace('-', '_')
                       .lower()
            )

        cleaned_colnames = [clean_colname(col) for col in df.columns]

        for orig, cleaned in zip(df.columns, cleaned_colnames):
            if df[orig].dtype == object:
                self.decode_property_labels(
                    encoded_prop_type=encoded_prop_type,
                    encoded_prop_name=orig,
                    new_prop_name=f"{cleaned}_decoded",
                    g=g,
                )
            else:
                print(f"{orig} prop left as is, no decoding needed (not an object type)")


    def get_category_code(self,
                        prop_name: str,
                        category_label: str,
                        dim: str = 'v') -> int:
        """
        Look up the integer code for a categorical property value. 
        This can then be used for lamda vertex (dim='v') or edge (dim='e') filtering.

        Parameters
        ----------
        prop_name : str
            The name of the property (vertex or edge) that is categorical.
        category_label : str
            The human-readable category label whose integer code you want.
        dim : str, optional
            'v' for a vertex property map, 'e' for an edge property map.

        Returns
        -------
        int
            The integer code for that category, or None if not present.
        """
        if dim == 'v':
            mapping_entry = self.core.vertex_categorical_mappings.get(prop_name)
        elif dim == 'e':
            mapping_entry = self.core.edge_categorical_mappings.get(prop_name)
        else:
            raise ValueError("dim must be 'v' or 'e'")

        if not mapping_entry or 'str_to_int' not in mapping_entry:
            raise KeyError(f"No categorical mapping found for {dim}-prop '{prop_name}'")

        cmap = mapping_entry['str_to_int']
        if category_label not in cmap:
            raise KeyError(f"Label '{category_label}' not seen for prop '{prop_name}'")
        return int(cmap[category_label])
