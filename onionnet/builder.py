from .core import OnionNetGraph
import pandas as pd
import numpy as np
from typing import List, Any
from .utils import infer_property_type, map_categorical_property

try:
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

#########################################
# Builder: Ingesting DataFrames into the Graph
#########################################
class OnionNetBuilder:
    def __init__(self, core: OnionNetGraph):
        self.core = core

    def grow_onion(
        self,
        df_nodes: pd.DataFrame,
        df_edges: pd.DataFrame,
        node_prop_cols: List[str] = None,
        edge_prop_cols: List[str] = None,
        drop_na: bool = True,
        drop_duplicates: bool = True,
        use_display: bool = False,
        node_id_col: str = 'node_id',
        node_layer_col: str = 'layer',
        edge_source_id_col: str = 'source_id',
        edge_source_layer_col: str = 'source_layer',
        edge_target_id_col: str = 'target_id',
        edge_target_layer_col: str = 'target_layer',
        vertex_property_types: dict = None,
        edge_property_types: dict = None
    ) -> None:
        # Use default property columns if none specified.
        node_prop_cols = node_prop_cols or ['node_prop_1', 'node_prop_2']
        edge_prop_cols = edge_prop_cols or ['edge_prop_1', 'edge_prop_2']
        
        # Validate required columns.
        missing_nodes = set([node_id_col, node_layer_col] + node_prop_cols) - set(df_nodes.columns)
        if missing_nodes:
            raise ValueError(f"Missing node columns: {missing_nodes}")
        missing_edges = set([edge_source_id_col, edge_source_layer_col,
                             edge_target_id_col, edge_target_layer_col] + edge_prop_cols) - set(df_edges.columns)
        if missing_edges:
            raise ValueError(f"Missing edge columns: {missing_edges}")
        
        if drop_duplicates:
            df_nodes = df_nodes.drop_duplicates(subset=[node_id_col, node_layer_col])
            df_edges = df_edges.drop_duplicates(
                subset=[edge_source_id_col, edge_source_layer_col, edge_target_id_col, edge_target_layer_col])
        
        # Display snippet and shape.
        for df, name in zip([df_nodes, df_edges], ['Nodes', 'Edges']):
            if use_display and IPYTHON_AVAILABLE:
                display(df.head())
                print(f"{name} shape: {df.shape}\n")
            else:
                pass
            
        
        self.add_vertices_from_dataframe(df_nodes, node_id_col, node_layer_col, node_prop_cols, drop_na, property_types=vertex_property_types)
        self.add_edges_from_dataframe(df_edges, edge_source_id_col, edge_source_layer_col,
                                      edge_target_id_col, edge_target_layer_col, edge_prop_cols, drop_na, property_types=edge_property_types)

    def add_vertices_from_dataframe(self, df_nodes: pd.DataFrame, id_col: str, layer_col: str,
                                    property_cols: List[str] = None, drop_na: bool = True,
                                    fill_na_with: Any = None, string_override: bool = False, property_types: dict = None) -> None:
        df = df_nodes.copy()
        if drop_na:
            df = df.dropna(subset=[id_col, layer_col])
        else:
            df = df.fillna({id_col: fill_na_with, layer_col: fill_na_with})
        
        # Map layers and node IDs.
        df['layer_int'] = df[layer_col].apply(self.core._map_layer)
        df['node_id_int'] = df[id_col].apply(self.core._map_node_id)
        custom_ids = list(zip(df['layer_int'], df['node_id_int']))
        n_new = len(custom_ids)
        start_idx = self.core.graph.num_vertices()
        self.core.graph.add_vertex(n_new)
        new_indices = np.arange(start_idx, start_idx + n_new, dtype=np.int64)
        self.core.custom_id_to_vertex_index.update(zip(custom_ids, new_indices))
        self.core.vertex_index_to_custom_id.update(zip(new_indices, custom_ids))
        
        # Bulk-assign core properties.
        self.core.graph.vp['layer_hash'].a[start_idx:] = df['layer_int'].values
        self.core.graph.vp['node_id_hash'].a[start_idx:] = df['node_id_int'].values
        
        # Assign additional properties.
        if property_cols:
            for prop in property_cols:
                values = df[prop].values
                if property_types and prop in property_types:
                    typ = property_types[prop]
                else:
                    typ = infer_property_type(df[prop])
                if typ in ['int', 'float'] and not string_override:
                    if prop not in self.core.graph.vp:
                        self.core.graph.vp[prop] = self.core.graph.new_vertex_property(typ)
                    self.core.graph.vp[prop].a[start_idx:] = values
                else:
                    mapped, mapping = map_categorical_property(prop, values)
                    self.core.vertex_categorical_mappings[prop] = {
                        'str_to_int': mapping,
                        'int_to_str': {v: k for k, v in mapping.items()}
                    }
                    if prop not in self.core.graph.vp:
                        self.core.graph.vp[prop] = self.core.graph.new_vertex_property('int')
                    self.core.graph.vp[prop].a[start_idx:] = mapped

    def add_edges_from_dataframe(self, df_edges: pd.DataFrame, source_id_col: str, source_layer_col: str,
                                 target_id_col: str, target_layer_col: str, property_cols: List[str] = None,
                                 drop_na: bool = True, fill_na_with: Any = None, string_override: bool = False, property_types: dict = None) -> None:
        df = df_edges.copy()
        if drop_na:
            df = df.dropna(subset=[source_id_col, source_layer_col, target_id_col, target_layer_col])
        else:
            df = df.fillna({source_id_col: fill_na_with, source_layer_col: fill_na_with,
                            target_id_col: fill_na_with, target_layer_col: fill_na_with})
        
        df['source_layer_int'] = df[source_layer_col].apply(self.core._map_layer)
        df['source_id_int'] = df[source_id_col].apply(self.core._map_node_id)
        df['target_layer_int'] = df[target_layer_col].apply(self.core._map_layer)
        df['target_id_int'] = df[target_id_col].apply(self.core._map_node_id)
        
        source_ids = list(zip(df['source_layer_int'], df['source_id_int']))
        target_ids = list(zip(df['target_layer_int'], df['target_id_int']))
        src_indices = [self.core.custom_id_to_vertex_index.get(t) for t in source_ids]
        tgt_indices = [self.core.custom_id_to_vertex_index.get(t) for t in target_ids]
        valid = [i for i, (s, t) in enumerate(zip(src_indices, tgt_indices)) if s is not None and t is not None]
        if not valid:
            print("No valid edges to add.")
            return
        edge_array = np.column_stack(([src_indices[i] for i in valid], [tgt_indices[i] for i in valid]))
        
        prop_list = []
        props = []
        if property_cols:
            for prop in property_cols:
                vals = df.iloc[valid][prop].values
                if property_types and prop in property_types:
                    typ = property_types[prop]
                else:
                    typ = infer_property_type(df[prop])
                if typ in ['int', 'float'] and not string_override:
                    if prop not in self.core.graph.ep:
                        self.core.graph.ep[prop] = self.core.graph.new_edge_property(typ)
                    prop_list.append(vals)
                    props.append(self.core.graph.ep[prop])
                else:
                    mapped, mapping = map_categorical_property(prop, vals)
                    self.core.edge_categorical_mappings[prop] = {
                        'str_to_int': mapping,
                        'int_to_str': {v: k for k, v in mapping.items()}
                    }
                    if prop not in self.core.graph.ep:
                        self.core.graph.ep[prop] = self.core.graph.new_edge_property('int')
                    prop_list.append(mapped)
                    props.append(self.core.graph.ep[prop])
        if prop_list:
            edge_list_with_props = np.column_stack((edge_array, *prop_list))
            self.core.graph.add_edge_list(edge_list_with_props, eprops=props)
        else:
            self.core.graph.add_edge_list(edge_array)