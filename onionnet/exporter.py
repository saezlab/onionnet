from __future__ import annotations

import pandas as pd

"""
This module provides export functionality for the OnionNetGraph.
It defines functions to export graph data (vertices and edges) to various formats such as a pandas DataFrame,
a list of dictionaries, or a dictionary keyed by IDs.
"""


def export_info(g, mode="v", prop_names=None, noisy=False, return_type="pandas"):
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

    try:
        import numpy as np
    except Exception:
        np = None

    if mode not in ("v", "e"):
        raise ValueError("mode must be 'v' (vertices) or 'e' (edges)")

    if mode == "v":
        prop_dict = g.vp
        items = g.vertices()
        base_keys = {"v_int"}
        def mk_id(v):
            return int(v)

        def base_builder(it):
            return {"v_int": mk_id(it)}
    else:
        prop_dict = g.ep
        items = g.edges()
        base_keys = {"e_id", "source", "target"}
        eid_map = g.edge_index  # works for Graph and GraphView
        def mk_id(e, _eid_map=eid_map):
            return int(_eid_map[e])

        def base_builder(it):
            return {
                "e_id": mk_id(it),
                "source": int(it.source()),
                "target": int(it.target()),
            }

    # Decide which props to export (never include built-ins)
    if prop_names is None:
        props = [p for p in prop_dict.keys() if p not in base_keys]
    else:
        requested = [p for p in prop_names if p not in base_keys]
        unknown = [p for p in requested if p not in prop_dict]
        if unknown:
            raise ValueError(f"Unknown {'edge' if mode == 'e' else 'vertex'} properties: {unknown}")
        props = requested

    rows = []
    for it in items:
        row = base_builder(it)
        for p in props:
            val = prop_dict[p][it]

            # --- normalize types for stable tests/serialization ---
            # numpy arrays / vector<double> -> list
            converted = False
            if hasattr(val, "tolist"):
                try:
                    val = val.tolist()
                    converted = True
                except Exception:
                    converted = False
            if not converted:
                # fallback: generic iterable (but not string/bytes)
                if isinstance(val, (list, tuple)):
                    pass  # already OK
                else:
                    try:
                        # treat as sequence if it is iterable and not a string-like
                        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                            val = list(val)
                    except Exception:
                        pass
            # numpy scalars -> Python scalars
            if np is not None and isinstance(val, np.generic):
                val = val.item()

            row[p] = val

        if noisy:
            if mode == "v":
                print(f"Vertex {row['v_int']}: " + ", ".join(f"{p} = {row[p]}" for p in props))
            else:
                print(
                    f"Edge {row['e_id']} ({row['source']} -> {row['target']}): "
                    + ", ".join(f"{p} = {row[p]}" for p in props)
                )
        rows.append(row)

    if return_type == "list":
        return rows
    elif return_type == "dict":
        key = "v_int" if mode == "v" else "e_id"
        return {r[key]: r for r in rows}
    elif return_type == "pandas":
        df = pd.DataFrame(rows)
        # Force Python ints for built-in edge columns so tests see `int`, not np.int64
        if mode == "e":
            for col in ("e_id", "source", "target"):
                if col in df.columns:
                    df[col] = df[col].map(int).astype(object)
        return df
    else:
        raise ValueError("Invalid return_type. Use 'list', 'dict', or 'pandas'.")
