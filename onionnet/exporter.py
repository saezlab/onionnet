from .core import OnionNetGraph
from graph_tool.all import Graph, GraphView
import pandas as pd
import numpy as np

#########################################
# Export Manager: From graph/filtered graph to list, csv, etc.
#########################################
import pandas as pd

def export_info(g, mode="v", prop_names: list = None, noisy: bool = False, return_type: str = "pandas"):
    """
    Export information from a Graph or GraphView. For vertices, it exports data from g.vp;
    for edges, it exports from g.ep along with source and target vertex IDs.
    
    Parameters
    ----------
    g : Graph or GraphView
        The graph whose data is to be exported.
    mode : str
        Specify "v" for vertices or "e" for edges.
    prop_names : list, optional
        A list of property names to include. If None, all properties from g.vp (or g.ep) will be used.
    noisy : bool, optional
        If True, print the info as it is processed.
    return_type : str, optional
        "pandas" (default) returns a DataFrame, "list" returns a list of dicts,
        "dict" returns a dict keyed by vertex/edge id.
    
    Returns
    -------
    Depending on return_type, returns a pandas DataFrame, a list of dictionaries, or a dictionary.
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
        # For edges we include source and target.
        def get_id(e):
            # If the graph has an edge_index, we use that.
            return int(g.edge_index[e]) if hasattr(g, "edge_index") else None
        items = g.edges()
        base_keys = ['e_id', 'source', 'target']
    else:
        raise ValueError("type must be 'v' (for vertices) or 'e' (for edges)")
    
    if prop_names is None:
        # Get all property keys from the appropriate map.
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