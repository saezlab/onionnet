import pandas as pd
import pytest

from onionnet.analytics import layer_stats, plot_layer_metagraph
from onionnet.builder import OnionNetBuilder
from onionnet.core import OnionNetGraph


def test_layer_stats_with_dataframes_and_interlayer():
    """
    layer_stats should compute node counts and edge-pair counts from DataFrames.
    If an 'interlayer' column exists, it is used for the interlayer count path
    (not returned, but exercised here).
    """
    df_nodes = pd.DataFrame(
        {
            "node_id": ["n1", "n2", "n3"],
            "layer": ["A", "A", "B"],
        },
    )
    df_edges = pd.DataFrame(
        {
            "source_id": ["n1", "n2", "n3"],
            "source_layer": ["A", "A", "B"],
            "target_id": ["n2", "n3", "n1"],
            "target_layer": ["A", "B", "A"],
            # mark interlayer explicitly for the 2 cross-layer edges
            "interlayer": [False, True, True],
        },
    )

    nodes_by_layer, edges_by_pair = layer_stats(
        df_nodes=df_nodes,
        df_edges=df_edges,
        print_tables=False,
    )

    # Node counts
    assert nodes_by_layer.loc["A", "count"] == 2
    assert nodes_by_layer.loc["B", "count"] == 1

    # Edge pair counts with MultiIndex (source_layer, target_layer)
    assert isinstance(edges_by_pair.index, pd.MultiIndex)
    got = {tuple(idx): int(val) for idx, val in edges_by_pair["edges"].items()}
    assert got == {("A", "A"): 1, ("A", "B"): 1, ("B", "A"): 1}


def test_layer_stats_from_core_only():
    """
    layer_stats should derive node counts from a built OnionNetGraph when
    DataFrames are not provided.
    """
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    df_nodes = pd.DataFrame(
        {
            "node_id": ["x", "y", "z"],
            "layer": ["X", "Y", "Y"],
        },
    )
    b.add_vertices_from_dataframe(df_nodes, id_col="node_id", layer_col="layer", drop_na=False)

    nodes_by_layer, edges_by_pair = layer_stats(core=core, print_tables=False)

    assert edges_by_pair is None  # no edge info available
    assert nodes_by_layer.loc["Y", "count"] == 2
    assert nodes_by_layer.loc["X", "count"] == 1


def test_layer_stats_missing_layer_raises():
    """
    If neither df_nodes has a layer column nor a core is provided, raise ValueError.
    """
    df_nodes = pd.DataFrame({"node_id": ["a", "b"]})
    with pytest.raises(ValueError):
        layer_stats(df_nodes=df_nodes, print_tables=False)


def test_plot_layer_metagraph_input_validation():
    """
    plot_layer_metagraph should validate both the presence of the 'edges' column
    and that the index is a MultiIndex of (source_layer, target_layer).
    """
    # Missing 'edges' column
    bad_df = pd.DataFrame({"foo": [1]}, index=pd.MultiIndex.from_tuples([("A", "B")]))
    with pytest.raises(ValueError):
        plot_layer_metagraph(bad_df, nodes_by_layer=None, show_labels=False)

    # Not a MultiIndex
    bad_idx_df = pd.DataFrame({"edges": [1]}, index=["A"])
    with pytest.raises(ValueError):
        plot_layer_metagraph(bad_idx_df, nodes_by_layer=None, show_labels=False)


def test_plot_layer_metagraph_builds_graph_and_calls_draw(monkeypatch):
    """
    With a valid edges_by_pair and optional nodes_by_layer, the function should
    build a meta-graph and call graph_draw with expected property maps.
    We monkeypatch graph_draw to avoid rendering and to inspect inputs.
    """
    # Prepare pair counts and nodes_by_layer
    pairs = pd.DataFrame(
        {"edges": [3, 1]},
        index=pd.MultiIndex.from_tuples(
            [("A", "B"), ("B", "B")],
            names=["source_layer", "target_layer"],
        ),
    )
    nodes = pd.DataFrame({"count": [5, 2]}, index=["A", "B"])

    captured = {}

    def fake_draw(g, **kwargs):
        _ = g
        captured["graph"] = g
        captured["kwargs"] = kwargs
        return

    # Monkeypatch the draw function used inside plot_layer_metagraph. The
    # function performs a local import `from graph_tool.all import ... graph_draw`,
    # so we patch graph_tool.all.graph_draw to intercept rendering.
    import graph_tool.all as gt_all

    monkeypatch.setattr(gt_all, "graph_draw", fake_draw)

    # Optionally use our own family extractor + colors to test coloring path
    fam_map = {"a": (1.0, 0.0, 0.0, 0.5), "b": (0.0, 0.0, 1.0, 0.5)}

    def fam_fn(s):
        return s.lower()

    res = plot_layer_metagraph(
        pairs,
        nodes_by_layer=nodes,
        show_labels=True,
        show_node_counts=True,
        show_edge_counts=True,
        use_monospace_font=True,
        family_colors=fam_map,
        family_extractor=fam_fn,
        return_graph=True,
    )

    # We requested return_graph
    assert isinstance(res, tuple) and len(res) == 2
    mg, pos = res
    assert mg.num_vertices() == 2  # layers A,B
    assert mg.num_edges() == 2  # A->B and B->B

    # graph_draw was called with maps we can inspect
    assert "kwargs" in captured and "vertex_fill_color" in captured["kwargs"]
    v_color = captured["kwargs"]["vertex_fill_color"]
    # Check that each vertex got a color matching its family
    # Map label -> color by reading the text property passed to draw
    v_text = captured["kwargs"]["vertex_text"]
    # v_text may be None if labels disabled; here we enabled labels
    assert v_text is not None

    # Build a mapping from label (first line) to its color
    lbl_to_col = {}
    for v in mg.vertices():
        label = v_text[v].split("\n")[0] if isinstance(v_text[v], str) else str(v_text[v])
        lbl_to_col[label] = tuple(v_color[v])

    assert lbl_to_col["A"] == fam_map["a"]
    assert lbl_to_col["B"] == fam_map["b"]

    # When edge counts are shown, an edge_text map should be supplied
    assert captured["kwargs"]["edge_text"] is not None


def test_plot_layer_metagraph_auto_colors_and_padding(monkeypatch):
    # Build pairs and minimal nodes_by_layer; rely on auto family colors and padding
    pairs = pd.DataFrame(
        {"edges": [2, 2]},
        index=pd.MultiIndex.from_tuples(
            [("alpha_u", "alpha_v"), ("beta_x", "beta_y")],
            names=["source_layer", "target_layer"],
        ),
    )
    nodes = pd.DataFrame({"count": [10, 4, 6, 2]}, index=["alpha_u", "alpha_v", "beta_x", "beta_y"])

    captured = {}

    def fake_draw(g, **kwargs):
        _ = g
        captured["kwargs"] = kwargs
        return

    import graph_tool.all as gt_all

    monkeypatch.setattr(gt_all, "graph_draw", fake_draw)

    plot_layer_metagraph(
        pairs,
        nodes_by_layer=nodes,
        show_labels=True,
        pad_label_string=True,
        show_node_counts=False,
        use_monospace_font=True,
        return_graph=False,
    )

    # Vertex text was provided
    assert captured["kwargs"].get("vertex_text") is not None


def test_plot_layer_metagraph_no_labels_and_empty_edges(monkeypatch):
    # Empty edges_by_pair with valid MultiIndex
    idx = pd.MultiIndex.from_tuples([], names=["source_layer", "target_layer"])
    empty_pairs = pd.DataFrame({"edges": []}, index=idx)

    called = {}

    def fake_draw(g, **kwargs):
        _ = g
        called["kwargs"] = kwargs
        return

    import graph_tool.all as gt_all

    monkeypatch.setattr(gt_all, "graph_draw", fake_draw)

    # No nodes_by_layer, no labels → go through default size path and vertex_text=None
    plot_layer_metagraph(
        empty_pairs,
        nodes_by_layer=None,
        show_labels=False,
        show_edge_counts=False,
    )
    assert called["kwargs"]["vertex_text"] is None


def test_layer_stats_edges_without_interlayer_and_uniform_weights():
    # Nodes in layers X and Y
    df_nodes = pd.DataFrame({"node_id": ["a", "b", "c", "d"], "layer": ["X", "X", "Y", "Y"]})
    # Edges: two X->Y edges (no 'interlayer' column)
    df_edges = pd.DataFrame(
        {
            "source_id": ["a", "b"],
            "source_layer": ["X", "X"],
            "target_id": ["c", "d"],
            "target_layer": ["Y", "Y"],
        },
    )

    nodes_by_layer, edges_by_pair = layer_stats(
        df_nodes=df_nodes,
        df_edges=df_edges,
        print_tables=False,
    )
    assert nodes_by_layer.loc["X", "count"] == 2 and nodes_by_layer.loc["Y", "count"] == 2
    # uniform weights: both edges are X->Y, so one pair with count=2
    assert edges_by_pair.loc[("X", "Y"), "edges"] == 2


def test_plot_layer_metagraph_uniform_edge_weights(monkeypatch):
    # Two edges with identical counts to trigger hi==lo branch in _scale
    pairs = pd.DataFrame(
        {"edges": [5, 5]},
        index=pd.MultiIndex.from_tuples(
            [("A", "B"), ("B", "A")],
            names=["source_layer", "target_layer"],
        ),
    )
    captured = {}

    def fake_draw(g, **kwargs):
        _ = g
        captured["kwargs"] = kwargs
        return

    import graph_tool.all as gt_all

    monkeypatch.setattr(gt_all, "graph_draw", fake_draw)

    plot_layer_metagraph(pairs, nodes_by_layer=None, show_labels=False, show_edge_counts=True)
    # edge_font_size provided when show_edge_counts True
    assert captured["kwargs"].get("edge_font_size") is not None


def test_layer_stats_edges_from_core_properties():
    import graph_tool.all as gt

    # Build a small graph with edge properties 'source_layer'/'target_layer' as integer codes
    g = gt.Graph(directed=True)
    g.add_vertex(3)
    e1 = g.add_edge(0, 1)
    e2 = g.add_edge(1, 2)
    # edge props as ints: 0->'L1', 1->'L2'
    ep_s = g.new_edge_property("int")
    ep_t = g.new_edge_property("int")
    ep_s[e1], ep_t[e1] = 0, 1
    ep_s[e2], ep_t[e2] = 1, 0
    g.ep["source_layer"] = ep_s
    g.ep["target_layer"] = ep_t

    class FakeCore:
        pass

    core = FakeCore()
    core.graph = g
    core.layer_code_to_name = {0: "L1", 1: "L2"}

    # Provide df_nodes for node counts; let edges_by_pair come from core.ep
    df_nodes = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["L1", "L2", "L1"]})
    nodes_by_layer, edges_by_pair = layer_stats(
        df_nodes=df_nodes,
        df_edges=None,
        core=core,
        print_tables=False,
    )
    # Expect pairs counted from core's edge props
    assert set(edges_by_pair.index.get_level_values(0)) | set(
        edges_by_pair.index.get_level_values(1),
    ) == {"L1", "L2"}


def test_layer_stats_print_tables(capsys):
    # Use small DataFrames and enable print_tables branch
    df_nodes = pd.DataFrame({"node_id": ["a", "b"], "layer": ["L", "L"]})
    df_edges = pd.DataFrame(
        {
            "source_id": ["a"],
            "source_layer": ["L"],
            "target_id": ["b"],
            "target_layer": ["L"],
        },
    )
    layer_stats(df_nodes=df_nodes, df_edges=df_edges, print_tables=True)
    out = capsys.readouterr().out
    assert "Node counts by layer" in out
