from .core import OnionNetGraph
import pandas as pd
import numpy as np
from typing import List, Any
from .utils import infer_property_type, map_categorical_property
import warnings

"""
This module provides the OnionNetBuilder class, which is responsible for ingesting node and edge DataFrames into an OnionNetGraph.
It handles data validation, duplicate removal, and mapping of properties for vertices and edges.
"""

try:
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

#########################################
# Builder: Ingesting DataFrames into the Graph
#########################################
class OnionNetBuilder:
    """
    Builder class for ingesting DataFrames into an OnionNetGraph.
    
    Attributes:
        core (OnionNetGraph): The core graph object where nodes and edges will be added.
    """
    def __init__(self, core: OnionNetGraph):
        """
        Initialize the OnionNetBuilder with a core OnionNetGraph instance.
        
        Parameters:
            core (OnionNetGraph): The core graph object used for adding vertices and edges.
        """
        self.core = core
        self._stats = {}

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
        edge_property_types: dict = None,
        verbose: bool = True,
    ) -> None:
        """
        Ingest node and edge DataFrames into the graph.
        
        This method validates the input DataFrames, and optionally displays
        a snippet of the data. By default, it then performs NA-dropping, duplicate removal,
        filters invalid edges, and finally adds vertices and edges to the graph.
        A summary of counts at each step is recorded and can be printed.
        
        Parameters:
            df_nodes (pd.DataFrame): DataFrame containing node information.
            df_edges (pd.DataFrame): DataFrame containing edge information.
            node_prop_cols (List[str], optional): List of node property column names. Defaults to None.
            edge_prop_cols (List[str], optional): List of edge property column names. Defaults to None.
            drop_na (bool, optional): Flag to drop rows with missing key values. Defaults to False.
            drop_duplicates (bool, optional): Flag to remove duplicate entries. Note this applies to _both_ nodes and edges. Defaults to False. 
            use_display (bool, optional): Flag to display a snippet of data using IPython display if available. Defaults to False.
            node_id_col (str, optional): Column name for node identifier. Defaults to 'node_id'.
            node_layer_col (str, optional): Column name for node layer information. Defaults to 'layer'.
            edge_source_id_col (str, optional): Column name for edge source identifier. Defaults to 'source_id'.
            edge_source_layer_col (str, optional): Column name for edge source layer information. Defaults to 'source_layer'.
            edge_target_id_col (str, optional): Column name for edge target identifier. Defaults to 'target_id'.
            edge_target_layer_col (str, optional): Column name for edge target layer information. Defaults to 'target_layer'.
            vertex_property_types (dict, optional): Mapping of vertex property types. Defaults to None.
            edge_property_types (dict, optional): Mapping of edge property types. Defaults to None.
            verbose (bool, optional): Prints summary of graph creation. Defaults to True.
        
        Raises:
            ValueError: If required columns are missing in the node or edge DataFrames.
        """
        # first, validate that all required columns exist
        if node_prop_cols is None:
            node_prop_cols = []
        if edge_prop_cols is None:
            edge_prop_cols = []
        missing_nodes = set([node_id_col, node_layer_col] + node_prop_cols) - set(df_nodes.columns)
        if missing_nodes:
            raise ValueError(f"Missing node columns: {missing_nodes}")
        missing_edges = set([
            edge_source_id_col, edge_source_layer_col,
            edge_target_id_col, edge_target_layer_col
        ] + edge_prop_cols) - set(df_edges.columns)
        if missing_edges:
            raise ValueError(f"Missing edge columns: {missing_edges}")

        # snapshot
        self._stats['nodes_in'] = len(df_nodes)
        self._stats['edges_in'] = len(df_edges)

        # copy once
        df_n = df_nodes.copy()
        df_e = df_edges.copy()

        # NA-dropping
        if drop_na:
            df_n_no_na = df_n.dropna(subset=[node_id_col, node_layer_col])
            df_e_no_na = df_e.dropna(subset=[edge_source_id_col, edge_source_layer_col,
                                             edge_target_id_col, edge_target_layer_col])
        else:
            # raise if any NA in keys
            if df_n[[node_id_col, node_layer_col]].isna().any().any():
                raise ValueError(f"NA in node keys but drop_na=False")
            if df_e[[edge_source_id_col, edge_source_layer_col,
                     edge_target_id_col, edge_target_layer_col]].isna().any().any():
                raise ValueError(f"NA in edge keys but drop_na=False")
            df_n_no_na = df_n
            df_e_no_na = df_e

        self._stats['nodes_dropped_na'] = len(df_nodes) - len(df_n_no_na)
        self._stats['edges_dropped_na'] = len(df_edges) - len(df_e_no_na)

        # 2) Duplicate removal
        if drop_duplicates:
            df_n_clean = df_n_no_na.drop_duplicates(subset=[node_id_col, node_layer_col])
            df_e_clean = df_e_no_na.drop_duplicates(
                subset=[edge_source_id_col, edge_source_layer_col,
                        edge_target_id_col, edge_target_layer_col])
        else:
            df_n_clean = df_n_no_na
            df_e_clean = df_e_no_na

        self._stats['nodes_deduped'] = len(df_n_no_na) - len(df_n_clean)
        self._stats['edges_deduped'] = len(df_e_no_na) - len(df_e_clean)

        # 3) Edge→node alignment: drop edges whose endpoints weren’t ingested
        #    we’ll let add_edges_from_dataframe do the final warning, but for stats:
        #    build the valid-edge index mask before passing to add_edges
        #    (note: add_edges_from_dataframe will repeat the int-mapping, so we just
        #     need the count here)
        # temporarily map through your core.custom_id_to_vertex_index dict
        src_pairs = list(zip(df_e_clean[edge_source_layer_col].astype(str),
                             df_e_clean[edge_source_id_col].astype(str)))
        tgt_pairs = list(zip(df_e_clean[edge_target_layer_col].astype(str),
                             df_e_clean[edge_target_id_col].astype(str)))
        src_idx = [self.core.custom_id_to_vertex_index.get(self.core._map_layer(l), 
                      self.core._map_node_id(i)) for l,i in src_pairs]
        tgt_idx = [self.core.custom_id_to_vertex_index.get(self.core._map_layer(l), 
                      self.core._map_node_id(i)) for l,i in tgt_pairs]
        valid_mask = [(s is not None and t is not None) for s,t in zip(src_idx, tgt_idx)]
        df_e_valid = df_e_clean[valid_mask]

        # record how many edges were invalidated by missing endpoints
        invalid_endpoints = len(df_e_clean) - len(df_e_valid)
        # total “invalid” = those dropped by NA + those dropped by bad endpoints
        self._stats['edges_dropped_invalid'] = (
            self._stats['edges_dropped_na']
            + invalid_endpoints
        )

        # optionally display
        if use_display and IPYTHON_AVAILABLE:
            display(df_n_clean.head()); print("Nodes →", df_n_clean.shape)
            display(df_e_valid.head());print("Edges →", df_e_valid.shape)

        # 4) Now actually add them
        self.add_vertices_from_dataframe(
            df_n_clean,
            id_col=node_id_col, layer_col=node_layer_col,
            property_cols=node_prop_cols,
            drop_na=False,                  # already cleaned
            drop_duplicates=False,          # already cleaned
            string_override=False,
            property_types=vertex_property_types
        )
        self.add_edges_from_dataframe(
            df_e_valid,
            source_id_col=edge_source_id_col, source_layer_col=edge_source_layer_col,
            target_id_col=edge_target_id_col, target_layer_col=edge_target_layer_col,
            property_cols=edge_prop_cols,
            drop_na=False,                  # already cleaned
            drop_duplicates=False,          # already cleaned
            string_override=False,
            property_types=edge_property_types
        )

        # final counts
        self._stats['nodes_final'] = self.core.graph.num_vertices()
        self._stats['edges_final'] = self.core.graph.num_edges()

        if verbose:
            print(self.summary())


    def add_vertices_from_dataframe(
        self,
        df_nodes: pd.DataFrame,
        id_col: str,
        layer_col: str,
        property_cols: List[str] = None,
        drop_na: bool = True,
        drop_duplicates: bool = True,
        string_override: bool = False,
        property_types: dict = None
    ) -> None:
        """
        Add vertices to the graph from a DataFrame containing node information.

        This method processes the DataFrame to ensure correct data types, handles missing values,
        maps node layers and IDs, and assigns both core and additional properties to the vertices.
        
        Parameters:
            df_nodes (pd.DataFrame): DataFrame containing node data.
            id_col (str): Name of the column containing node identifiers.
            layer_col (str): Name of the column containing node layer information.
            property_cols (List[str], optional): List of additional node property columns.
            drop_na (bool): Drop rows with missing id/layer if True; else error on NA.
            drop_duplicates (bool): Drop duplicate (id,layer) rows if True.
            string_override (bool): Treat all props as categorical if True.
            property_types (dict): Explicit types for properties.
        """
        # check for column name conflicts with internal keys
        internal = {'layer_hash','node_id_hash','v_int'}
        if property_cols:
            collision = set(property_cols) & internal
            if collision:
                raise ValueError(f"Cannot use {collision!r} as property name, it's reserved for internal keys")
        # prop-col existence
        if property_cols:
            missing = set(property_cols) - set(df_nodes.columns)
            if missing:
                raise ValueError(f"Property columns not found in nodes DataFrame: {missing}")
        df = df_nodes.copy()
        # drop_na enforcement
        if drop_na:
            df = df.dropna(subset=[id_col, layer_col])
        else:
            if df[[id_col, layer_col]].isna().any().any():
                raise ValueError(
                    f"Detected NA in {id_col}/{layer_col} but drop_na=False; "
                    "please set drop_na=True or clean your data first."
                )
        # warn if duplicates would be kept
        if not drop_duplicates:
            ndup = df.duplicated(subset=[id_col, layer_col]).sum()
            if ndup:
                warnings.warn(
                    f"{ndup} duplicate node rows found but drop_duplicates=False",
                    UserWarning
                )
        # duplicate removal
        if drop_duplicates:
            df = df.drop_duplicates(subset=[id_col, layer_col])
        # cast keys
        df[id_col]   = df[id_col].astype(str)
        df[layer_col] = df[layer_col].astype(str)
        # map ints
        df['layer_int']    = df[layer_col].apply(self.core._map_layer)
        df['node_id_int']  = df[id_col].apply(self.core._map_node_id)
        custom_ids = list(zip(df['layer_int'], df['node_id_int']))
        n_new = len(custom_ids)
        start = self.core.graph.num_vertices()
        self.core.graph.add_vertex(n_new)
        # new_idx = np.arange(start, start+n_new, dtype=np.int64) #<--- previously
        # use Python ints so downstream code/test isinstance(idx, int) passes and for more numerical stability
        new_idx = list(range(start, start+n_new))
        self.core.custom_id_to_vertex_index.update(zip(custom_ids, new_idx))
        self.core.vertex_index_to_custom_id.update(zip(new_idx, custom_ids))
        # bulk core props
        self.core.graph.vp['layer_hash'].a[start:]   = df['layer_int'].values
        self.core.graph.vp['node_id_hash'].a[start:] = df['node_id_int'].values
        # additional props
        if property_cols:
            for prop in property_cols:
                vals = df[prop].values
                typ = property_types.get(prop) if property_types and prop in property_types else infer_property_type(df[prop])
                if typ in ['int','float'] and not string_override:
                    if prop not in self.core.graph.vp:
                        self.core.graph.vp[prop] = self.core.graph.new_vertex_property(typ)
                    self.core.graph.vp[prop].a[start:] = vals
                else:
                    # extend existing mapping instead of restarting it
                    if prop in self.core.vertex_categorical_mappings:
                        cmap = self.core.vertex_categorical_mappings[prop]['str_to_int']
                        inv  = self.core.vertex_categorical_mappings[prop]['int_to_str']
                        mapped = []
                        for v in vals:
                            if v not in cmap:
                                newcode = max(cmap.values(), default=-1) + 1
                                cmap[v]   = newcode
                                inv[newcode] = v
                            mapped.append(cmap[v])
                    else:
                        mapped, mapping = map_categorical_property(prop, vals)
                        inv = {v:k for k,v in mapping.items()}
                        self.core.vertex_categorical_mappings[prop] = {
                            'str_to_int': mapping,
                            'int_to_str': inv
                        }

                    # ensure the vp exists and assign
                    if prop not in self.core.graph.vp:
                        self.core.graph.vp[prop] = self.core.graph.new_vertex_property('int')
                    self.core.graph.vp[prop].a[start:] = np.array(mapped, dtype=int)


    def add_edges_from_dataframe(
        self,
        df_edges: pd.DataFrame,
        source_id_col: str,
        source_layer_col: str,
        target_id_col: str,
        target_layer_col: str,
        property_cols: List[str] = None,
        drop_na: bool = True,
        drop_duplicates: bool = True,
        string_override: bool = False,
        property_types: dict = None,
        consider_props_in_duplicate: bool = False,
    ) -> None:
        """
        Add edges to the graph from a DataFrame containing edge information.

        This method processes the DataFrame to ensure correct data types, handles missing values,
        maps source and target node identifiers, and assigns properties to the edges.
        
        Parameters:
            df_edges (pd.DataFrame): DataFrame containing edge data.
            source_id_col, source_layer_col, target_id_col, target_layer_col: key columns.
            property_cols (List[str], optional): List of edge property columns.
            drop_na (bool): Drop rows with missing keys if True; else error on NA.
            drop_duplicates (bool): Drop duplicate edge pairs (source,layer,target,layer rows) if True, including double self loops.
            string_override (bool): Treat all props as categorical.
            property_types (dict): Explicit types for properties.
            consider_props_in_duplicate (bool): If True, duplicates are defined by endpoints and the listed property columns (so two nodes could be from A1->A2, but with different properties or property values), otherwise only by endpoints.
        """
        # if there are literally no rows to add, bail out immediately
        if df_edges.shape[0] == 0:
            return
        # check for column name conflicts with internal keys
        internal = {'e_id','source','target'}  
        if property_cols:
            collision = set(property_cols) & internal
            if collision:
                raise ValueError(f"Cannot use {collision!r} as property name, it's reserved for internal keys")
        # prop-col existence
        if property_cols:
            missing = set(property_cols) - set(df_edges.columns)
            if missing:
                raise ValueError(f"Property columns not found in edges DataFrame: {missing}")
        df = df_edges.copy()
        # NA handling
        if drop_na:
            df = df.dropna(subset=[source_id_col, source_layer_col, target_id_col, target_layer_col])
        else:
            if df[[source_id_col, source_layer_col, target_id_col, target_layer_col]].isna().any().any():
                raise ValueError(
                    f"Detected NA in edge keys but drop_na=False; "
                    "please set drop_na=True or clean your data first."
                )
        uniq_subset = [
            source_id_col, source_layer_col,
            target_id_col, target_layer_col
        ]
        if drop_duplicates:
            if consider_props_in_duplicate and property_cols:
                uniq_subset += property_cols
            df = df.drop_duplicates(subset=uniq_subset)
        else: # warn if duplicates would be kept
            ndup = df.duplicated(
                subset=[source_id_col, source_layer_col, target_id_col, target_layer_col]
            ).sum()
            if ndup:
                warnings.warn(
                    f"{ndup} duplicate edge rows found but drop_duplicates=False",
                    UserWarning
                )
        # cast keys
        df[source_id_col] = df[source_id_col].astype(str)
        df[source_layer_col] = df[source_layer_col].astype(str)
        df[target_id_col] = df[target_id_col].astype(str)
        df[target_layer_col] = df[target_layer_col].astype(str)
        # map ints
        df['source_layer_int'] = df[source_layer_col].apply(self.core._map_layer)
        df['source_id_int']    = df[source_id_col].apply(self.core._map_node_id)
        df['target_layer_int'] = df[target_layer_col].apply(self.core._map_layer)
        df['target_id_int']    = df[target_id_col].apply(self.core._map_node_id)
        source_ids = list(zip(df['source_layer_int'], df['source_id_int']))
        target_ids = list(zip(df['target_layer_int'], df['target_id_int']))
        src_idx = [self.core.custom_id_to_vertex_index.get(t) for t in source_ids]
        tgt_idx = [self.core.custom_id_to_vertex_index.get(t) for t in target_ids]
        valid = [i for i, (s,t) in enumerate(zip(src_idx, tgt_idx)) if s is not None and t is not None]
        if not valid:
            warnings.warn(
                "No valid edges to add: all edges reference missing vertices.", UserWarning
            )
            return
        edge_arr = np.column_stack((
            [src_idx[i] for i in valid],
            [tgt_idx[i] for i in valid]
        ))
        # properties
        prop_vals = []
        prop_maps = []
        if property_cols:
            for prop in property_cols:
                vals = df.iloc[valid][prop].values
                typ = property_types.get(prop) if property_types and prop in property_types else infer_property_type(df[prop])
                if typ in ['int','float'] and not string_override:
                    # numeric, existing code untouched
                    if prop not in self.core.graph.ep:
                        self.core.graph.ep[prop] = self.core.graph.new_edge_property(typ)
                    prop_vals.append(vals)
                    prop_maps.append(self.core.graph.ep[prop])
                else:
                    # —— begin categorical extension logic —— #
                    if prop in self.core.edge_categorical_mappings:
                        cmap = self.core.edge_categorical_mappings[prop]['str_to_int']
                        inv  = self.core.edge_categorical_mappings[prop]['int_to_str']
                        mapped = []
                        for v in vals:
                            if v not in cmap:
                                code = max(cmap.values(), default=-1) + 1
                                cmap[v]   = code
                                inv[code] = v
                            mapped.append(cmap[v])
                    else:
                        mapped, mapping = map_categorical_property(prop, vals)
                        inv = {v: k for k,v in mapping.items()}
                        self.core.edge_categorical_mappings[prop] = {
                            'str_to_int': mapping,
                            'int_to_str': inv
                        }

                    # make sure the edge‐property exists as an int property
                    if prop not in self.core.graph.ep:
                        self.core.graph.ep[prop] = self.core.graph.new_edge_property('int')

                    prop_vals.append(np.array(mapped, dtype=int))
                    prop_maps.append(self.core.graph.ep[prop])

        # now actually add into the graph
        if prop_vals:
            # stack the (src,tgt) with each prop‐column’s values
            arr = np.column_stack((edge_arr, *prop_vals))
            self.core.graph.add_edge_list(arr, eprops=prop_maps)
        else:
            # no extra properties → just add the bare edges
            self.core.graph.add_edge_list(edge_arr)


    def summary(self) -> str:
            s = self._stats
            return (
                f"Nodes: in={s['nodes_in']}, dropped_na={s['nodes_dropped_na']}, "
                f"deduped={s['nodes_deduped']} → final={s['nodes_final']}\n"
                f"Edges: in={s['edges_in']}, dropped_invalid={s['edges_dropped_invalid']}, "
                f"deduped={s['edges_deduped']} → final={s['edges_final']}"
            )