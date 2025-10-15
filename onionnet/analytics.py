"""Analytics utilities for OnionNet graphs.

Provide layer statistics and a meta-graph plotting helper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Optional: only import if available in your env
try:
    from IPython.display import display
except Exception:
    display = print  # fallback: just print DataFrames


# --- add this helper (replaces the hard-coded _default_family) ---
def _infer_family_basic(name: str) -> str:
    """Infer a coarse 'family' from a layer name.

    Heuristic: take the lowercase prefix before the first separator.
    Separators checked (in order): '_', ':', '/', '-', space.
    """
    n = (name or "").strip().lower()
    for sep in ("_", ":", "/", "-", " "):
        if sep in n:
            return n.split(sep, 1)[0]
    return n


if TYPE_CHECKING:
    # For static analysis; avoids runtime import cycles
    from collections.abc import Callable

    from graph_tool.all import Graph, PropertyMap

    from .core import OnionNetGraph


def layer_stats(
    df_nodes: pd.DataFrame | None = None,
    df_edges: pd.DataFrame | None = None,
    *,
    core: OnionNetGraph | None = None,
    node_layer_col: str = "layer",
    source_layer_col: str = "source_layer",
    target_layer_col: str = "target_layer",
    print_tables: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Compute quick layer summaries.

    You can pass DataFrames (fast, no need to scan the graph) OR, if you
    already built the graph and kept layer props on vertices/edges, pass
    `core` and it will derive the same tables.

    Returns
    -------
    nodes_by_layer : DataFrame with a single 'count' column (index = layer)
    interlayer_edge_count : int or None
    edges_by_pair : DataFrame with a single 'edges' column
    """
    # --- nodes_by_layer ---
    if df_nodes is not None and node_layer_col in df_nodes:
        nodes_by_layer = (
            df_nodes[node_layer_col]
            .value_counts(dropna=False)
            .sort_values(ascending=False)
            .to_frame("count")
        )
    elif core is not None and "layer_decoded" in core.graph.vp:
        vals = [core.graph.vp["layer_decoded"][v] for v in core.graph.vertices()]
        nodes_by_layer = (
            pd.Series(vals)
            .value_counts(dropna=False)
            .sort_values(ascending=False)
            .to_frame("count")
        )
    elif core is not None and "layer_hash" in core.graph.vp and hasattr(core, "layer_code_to_name"):
        vals = [
            core.layer_code_to_name.get(int(core.graph.vp["layer_hash"][v]))
            for v in core.graph.vertices()
        ]
        nodes_by_layer = (
            pd.Series(vals)
            .value_counts(dropna=False)
            .sort_values(ascending=False)
            .to_frame("count")
        )
    else:
        raise ValueError("Need df_nodes[layer] or vertex layer props on core.graph.")

    # --- interlayer + edges_by_pair (if edges available) ---
    interlayer_edge_count = None
    edges_by_pair = None

    if df_edges is not None and {source_layer_col, target_layer_col}.issubset(df_edges.columns):
        # compute counts from the DataFrame directly
        if "interlayer" in df_edges:
            interlayer_edge_count = int(df_edges["interlayer"].sum())
        else:
            interlayer_edge_count = int(
                (df_edges[source_layer_col] != df_edges[target_layer_col]).sum(),
            )

        edges_by_pair = (
            df_edges.groupby([source_layer_col, target_layer_col])
            .size()
            .sort_values(ascending=False)
            .to_frame("edges")
        )

    elif core is not None:
        g = core.graph
        # we can compute from graph if source/target layer edge props exist
        if ("source_layer" in g.ep) and ("target_layer" in g.ep):
            s = [g.ep["source_layer"][e] for e in g.edges()]
            t = [g.ep["target_layer"][e] for e in g.edges()]
            # decode if necessary
            if isinstance(s[0], int | np.integer):
                s = [core.layer_code_to_name.get(int(x)) for x in s]
            if isinstance(t[0], int | np.integer):
                t = [core.layer_code_to_name.get(int(x)) for x in t]

            ser = pd.Series(list(zip(s, t, strict=False)))
            edges_by_pair = ser.value_counts().sort_values(ascending=False).to_frame("edges")
            edges_by_pair.index = pd.MultiIndex.from_tuples(
                edges_by_pair.index,
                names=[source_layer_col, target_layer_col],
            )
            interlayer_edge_count = int((pd.Series(s) != pd.Series(t)).sum())

    if print_tables:
        print("\nNode counts by layer:")
        display(nodes_by_layer)
        if interlayer_edge_count is not None:
            print("\nInterlayer edge count:", interlayer_edge_count)
        if edges_by_pair is not None:
            print("Edge counts by (source_layer, target_layer):")
            display(edges_by_pair)

    # deprecated return of interlayer_edge_count
    return nodes_by_layer, edges_by_pair


def plot_layer_metagraph(
    edges_by_pair: pd.DataFrame,
    nodes_by_layer: pd.DataFrame | None = None,
    *,
    # scaling of geometry
    node_scaler: str = "log",  # {'log','linear'}
    edge_scaler: str = "log",  # {'log','linear'}
    node_size_range: tuple[float, float] = (10, 60),
    edge_width_range: tuple[float, float] = (0.5, 8.0),
    # text sizes
    node_text_size_range: tuple[float, float] = (10, 16),
    edge_text_size_range: tuple[float, float] = (8, 14),
    # labels
    show_edge_counts: bool = False,
    show_node_counts: bool = False,
    node_label_fmt: str = "{layer}\n(n={count})",
    node_text_position: int | float = -1,  # -1=center inside; or angle in radians
    # pad node labels to same length (first line only)
    pad_label_string: bool = False,
    # monospace font toggle + optional explicit font names
    use_monospace_font: bool = False,
    vertex_font: str | None = None,
    edge_font: str | None = None,
    # color + layout
    family_colors: dict[str, tuple[float, float, float, float]] | None = None,
    family_extractor: Callable[[str], str] | None = None,
    layout: str = "sfdp",
    show_labels: bool = True,
    output_size: tuple[int, int] = (900, 700),
    return_graph: bool = False,
    # custom positioning (reuse prior layout)
    pos: PropertyMap | None = None,
) -> tuple[Graph, PropertyMap] | None:
    """
    Draw a meta-graph whose vertices are layers and edges count cross-layer edges.

    What it shows
    -------------
    • Vertex size ∝ per-layer node count (uniform if `nodes_by_layer=None`).
    • Edge width ∝ edges between layer pairs.
    • Optional labels:
        - `show_node_counts=True`: include per-layer count in the vertex label.
        - `show_edge_counts=True`: draw the edge count as a label on edges.

    Scaling & ranges
    ----------------
    • Counts are transformed by `node_scaler` / `edge_scaler` ('log' or 'linear').
    • Clamped to `node_size_range`, `edge_width_range`.
    • Text sizes are independently clamped via `node_text_size_range`, `edge_text_size_range`.

    Parameters
    ----------
    edges_by_pair : DataFrame
        MultiIndex (source_layer, target_layer) with an 'edges' column.
    nodes_by_layer : DataFrame or None
        Optional per-layer node counts (column 'count').
    node_text_position : float|int
        `-1` to center text inside node; otherwise angle (radians) relative to node.
    family_colors : dict or None
        Map family→RGBA; overrides auto palette. Families are derived by `family_extractor`
        (or a simple prefix heuristic if None).
    family_extractor : callable or None
        Given a layer name → family string (used for coloring).
    pos : VertexPropertyMap or None
        Reuse a precomputed layout; if None, compute one.

    Returns
    -------
    (Graph, PropertyMap) or None
        Returns the constructed meta-graph and its layout if ``return_graph=True``,
        otherwise returns ``None``.
    """
    # Local imports so tests can monkeypatch and to avoid heavy deps at import-time
    from graph_tool.all import Graph, graph_draw, sfdp_layout
    import matplotlib.cm as cm
    import numpy as np

    if not isinstance(edges_by_pair, pd.DataFrame) or "edges" not in edges_by_pair.columns:
        raise ValueError(
            "edges_by_pair must be a DataFrame with an 'edges' column and MultiIndex of (source_layer, target_layer).",
        )
    if not isinstance(edges_by_pair.index, pd.MultiIndex):
        raise ValueError(
            "edges_by_pair index must be a MultiIndex of (source_layer, target_layer).",
        )

    def _scale(vals: np.ndarray, mode: str, out_range: tuple[float, float]) -> np.ndarray:
        if vals.size == 0:
            return vals
        x = np.asarray(vals, dtype=float)
        if mode == "log":
            x = np.log1p(np.maximum(x, 0.0))
        lo, hi = float(x.min()), float(x.max())
        if hi == lo:
            return np.full_like(x, np.mean(out_range), dtype=float)
        t = (x - lo) / (hi - lo)
        return out_range[0] + t * (out_range[1] - out_range[0])

    # Default family extractor (prefix before one of common separators)
    def _infer_family_basic(name: str) -> str:
        n = (str(name) or "").strip().lower()
        for sep in ("_", ":", "/", "-", " "):
            if sep in n:
                return n.split(sep, 1)[0]
        return n

    layers = sorted(
        set(edges_by_pair.index.get_level_values(0)) | set(edges_by_pair.index.get_level_values(1)),
    )

    mg = Graph(directed=True)
    v_map: dict[str, int] = {}

    v_label = mg.new_vertex_property("string")
    v_family = mg.new_vertex_property("string")
    v_color = mg.new_vertex_property("vector<double>")
    v_size = mg.new_vertex_property("double")
    v_fsize = mg.new_vertex_property("double")  # node text size

    fam_fn = family_extractor or _infer_family_basic

    for layer in layers:
        v = mg.add_vertex()
        v_map[layer] = int(v)
        v_label[v] = str(layer)
        v_family[v] = fam_fn(layer)

    # Colors
    if family_colors:
        palette = dict(family_colors)
        default_rgba = (0.40, 0.40, 0.40, 0.85)
        for v in mg.vertices():
            v_color[v] = palette.get(v_family[v], default_rgba)
    else:
        fams = sorted({v_family[v] for v in mg.vertices()})
        cmap = cm.get_cmap("tab20")
        fam_to_rgba = {f: tuple(cmap(i / max(1, len(fams) - 1))) for i, f in enumerate(fams)}
        # unify alpha
        fam_to_rgba = {f: (rgba[0], rgba[1], rgba[2], 0.9) for f, rgba in fam_to_rgba.items()}
        for v in mg.vertices():
            v_color[v] = fam_to_rgba[v_family[v]]

    # Node sizes & text sizes
    layer_counts: dict[str, int] = {}
    if nodes_by_layer is not None and "count" in nodes_by_layer.columns:
        layer_counts = {str(idx): int(val) for idx, val in nodes_by_layer["count"].items()}
        arr = np.array([layer_counts.get(v_label[v], 1) for v in mg.vertices()], dtype=float)
        sizes = _scale(arr, node_scaler, node_size_range)
        fsize = _scale(arr, node_scaler, node_text_size_range)
    else:
        arr = np.ones(mg.num_vertices(), dtype=float)
        sizes = np.full_like(arr, np.mean(node_size_range), dtype=float)
        fsize = np.full_like(arr, np.mean(node_text_size_range), dtype=float)

    for vv, s, fs in zip(mg.vertices(), sizes, fsize, strict=False):
        v_size[vv] = float(s)
        v_fsize[vv] = float(fs)

    # Edges
    e_weight = mg.new_edge_property("double")
    e_width = mg.new_edge_property("double")
    e_color = mg.new_edge_property("vector<double>")
    e_text = mg.new_edge_property("string")
    e_fsize = mg.new_edge_property("double")

    edge_rgba = (0.2, 0.2, 0.2, 0.45)
    pairs = edges_by_pair.sort_values("edges", ascending=False)
    weights = pairs["edges"].astype(float).values
    widths = _scale(weights, edge_scaler, edge_width_range)
    tsize = _scale(weights, edge_scaler, edge_text_size_range)

    for (src, tgt), w, pen, tfs in zip(pairs.index.values, weights, widths, tsize, strict=False):
        e = mg.add_edge(v_map[str(src)], v_map[str(tgt)])
        e_weight[e] = float(w)
        e_width[e] = float(pen)
        e_color[e] = edge_rgba
        if show_edge_counts:
            e_text[e] = f"{int(w):,}"  # nice thousands separators
            e_fsize[e] = float(tfs)

    # Vertex labels (optionally with counts)
    if show_labels:
        if show_node_counts:
            for v in mg.vertices():
                lbl = str(v_label[v])
                cnt = layer_counts.get(lbl, 0)
                v_label[v] = node_label_fmt.format(layer=lbl, count=f"{cnt:,}")
        # pad first line if requested (works best with monospace)
        if pad_label_string:
            labels = [str(v_label[v]) for v in mg.vertices()]
            first_lines = [(s.splitlines()[0] if "\n" in s else s) for s in labels]
            max_len = max((len(fl) for fl in first_lines), default=0)
            v_label_padded = mg.new_vertex_property("string")
            for v in mg.vertices():
                s = str(v_label[v])
                parts = s.splitlines()
                parts[0] = parts[0].ljust(max_len)
                v_label_padded[v] = "\n".join(parts)
            v_text = v_label_padded
        else:
            v_text = v_label
    else:
        v_text = None

    # Layout: keep user-supplied pos if given
    if pos is None:
        # Choose layout based on argument (currently only 'sfdp' available)
        if layout == "sfdp":
            pos = sfdp_layout(mg)
        else:
            raise ValueError(f"Unknown layout '{layout}'.")

    # Fonts
    v_font = (
        vertex_font if vertex_font is not None else ("Monospace" if use_monospace_font else None)
    )
    e_font = edge_font if edge_font is not None else ("Monospace" if use_monospace_font else None)

    graph_draw(
        mg,
        pos=pos,
        vertex_text=v_text,
        vertex_text_position=node_text_position,
        vertex_font_size=v_fsize,
        vertex_font_family=v_font,
        vertex_size=v_size,
        vertex_fill_color=v_color,
        edge_pen_width=e_width,
        edge_color=e_color,
        edge_text=(e_text if show_edge_counts else None),
        edge_text_parallel=True,
        edge_font_size=(e_fsize if show_edge_counts else None),
        edge_font_family=(e_font if show_edge_counts else None),
        output_size=output_size,
    )

    if return_graph:
        return mg, pos
    return None
