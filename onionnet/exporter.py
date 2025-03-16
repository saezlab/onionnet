from .core import OnionNetGraph
from graph_tool.all import Graph, GraphView
import pandas as pd
import numpy as np

"""
This module provides export functionality for the OnionNetGraph.
It defines functions to export graph data (vertices and edges) to various formats such as a pandas DataFrame,
a list of dictionaries, or a dictionary keyed by IDs.
"""


def export_info(g, mode="v", prop_names: list = None, noisy: bool = False, return_type: str = "pandas"):
    """
    Export information from a graph (Graph or GraphView) into a structured format.
    
    This function extracts properties from either vertices or edges based on the specified mode.
    For vertices, it uses the vertex properties (g.vp) and for edges, it uses edge properties (g.ep)
    along with source and target vertex identifiers.
    
    Parameters
    ----------
    g : Graph or GraphView
        The graph from which to export data.
    mode : str, optional
        Export mode: 'v' for vertices, 'e' for edges. Default is 'v'.
    prop_names : list, optional
        A list of property names to include in the export. If None, all properties from g.vp (or g.ep) will be used.
    noisy : bool, optional
        If True, print details of the exported data during processing. Default is False.
    return_type : str, optional
        The format of the returned data:
          - "pandas" (default) returns a pandas DataFrame
          - "list" returns a list of dictionaries
          - "dict" returns a dictionary keyed by vertex or edge ID
    
    Returns
    -------
    pandas.DataFrame, list, or dict
        The exported graph information in the requested format.
    
    Raises
    ------
    ValueError
        If the mode is not 'v' or 'e', or if an invalid return_type is specified.
    """
    if mode == "v":
        # Export vertex information using g.vp
        prop_map = g.vp
        get_id = lambda v: int(v)
        items = g.vertices()
        base_keys = ['v_int']
    elif mode == "e":
        # Export edge information using g.ep
        prop_map = g.ep
        # For edges, include source and target vertices.
        def get_id(e):
            # If the graph has an edge_index, use it to get the edge ID.
            return int(g.edge_index[e]) if hasattr(g, "edge_index") else None
        items = g.edges()
        base_keys = ['e_id', 'source', 'target']
    else:
        raise ValueError("mode must be 'v' (for vertices) or 'e' (for edges)")
    
    if prop_names is None:
        # Use all property keys from the property map.
        prop_names = list(prop_map.keys())
    
    info_list = []
    for item in items:
        if mode == "v":
            info = {'v_int': get_id(item)}
        else:
            info = {
                'e_id': get_id(item),
                'source': int(item.source()),
                'target': int(item.target())
            }
        for prop in prop_names:
            info[prop] = prop_map[prop][item]
        if noisy:
            props_str = ", ".join(f"{prop} = {info[prop]}" for prop in prop_names)
            if mode == "v":
                print(f"Vertex {info['v_int']}: {props_str}")
            else:
                print(f"Edge {info['e_id']} ({info['source']} -> {info['target']}): {props_str}")
        info_list.append(info)
    
    if return_type == "list":
        return info_list
    elif return_type == "dict":
        # Keyed by the ID (vertex or edge)
        key_name = "v_int" if mode == "v" else "e_id"
        return {item[key_name]: item for item in info_list}
    elif return_type == "pandas":
        return pd.DataFrame(info_list)
    else:
        raise ValueError("Invalid return_type. Use 'list', 'dict', or 'pandas'.")