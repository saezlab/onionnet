import pytest
import numpy as np
import pandas as pd
from graph_tool.all import GraphView, shortest_distance

from onionnet.core import OnionNetGraph
from onionnet.builder import OnionNetBuilder
from onionnet.searcher import OnionNetSearcher
from onionnet.exporter import export_info


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def simple_graph():
    """
    Build a tiny 2-node, 1-edge graph with a numeric vertex property 'score'.
    """
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"], "score": [10, 20]})
    b.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=["score"], drop_na=False)
    # connect A->B so that edges exist
    b.add_edges_from_dataframe(
        pd.DataFrame(
            {
                "source_id": ["A"],
                "source_layer": ["0"],
                "target_id": ["B"],
                "target_layer": ["0"],
            }
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    return core


@pytest.fixture
def builder_and_core():
    """
    Provide a fresh builder and core graph for tests.
    """
    core = OnionNetGraph()
    builder = OnionNetBuilder(core)
    return builder, core


@pytest.fixture
def chain_graph_and_searcher():
    """
    Build a simple directed chain A→B→C, and return both core and its searcher.
    """
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    # nodes A,B,C
    df_nodes = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"]})
    b.add_vertices_from_dataframe(df_nodes, "node_id", "layer", drop_na=False)
    # edges A→B, B→C
    df_edges = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "C"],
            "target_layer": ["0", "0"],
        }
    )
    b.add_edges_from_dataframe(
        df_edges,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    searcher = OnionNetSearcher(core)
    return core, searcher


# --- 1. Vertex export to pandas --------------------------------------------


def test_export_vertices_to_pandas(builder_and_core, simple_graph):
    """
    When exporting vertices without specifying prop_names, all vp keys appear;
    default return_type='pandas' yields a DataFrame.
    """
    # add an extra categorical prop 'grp'
    builder, core = builder_and_core
    df = pd.DataFrame(
        {"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"], "grp": ["x", "y", "x"]}
    )
    builder.add_vertices_from_dataframe(
        df, "node_id", "layer", property_cols=["grp"], drop_na=False
    )

    df_out = export_info(core.graph, mode="v", return_type="pandas")
    # must include 'v_int', 'layer_hash', 'node_id_hash', plus 'grp'
    assert "v_int" in df_out.columns
    assert "grp" in df_out.columns
    # row count matches number of vertices
    assert len(df_out) == core.graph.num_vertices()


# --- 2. Edge export to list of dicts ---------------------------------------


def test_export_edges_to_list(builder_and_core):
    """
    Exporting edges with return_type='list' yields a list of dicts containing
    'e_id', 'source', 'target', and any specified properties.
    """
    builder, core = builder_and_core
    # seed two nodes and one edge with strength
    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]})
    builder.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A"],
            "source_layer": ["0"],
            "target_id": ["B"],
            "target_layer": ["0"],
            "strength": [42],
        }
    )
    builder.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["strength"],
        drop_na=False,
    )

    lst = export_info(core.graph, mode="e", return_type="list")
    # should be a list with one dict
    assert isinstance(lst, list) and len(lst) == 1
    entry = lst[0]
    assert set(entry.keys()) == {"e_id", "source", "target", "strength"}
    assert entry["strength"] == 42
    assert entry["source"] == 0 and entry["target"] == 1


# --- 3. Edge export to dict keyed by e_id ----------------------------------


def test_export_edges_to_dict(builder_and_core):
    """
    return_type='dict' should key the output by edge ID.
    """
    builder, core = builder_and_core
    # seed two nodes + two edges
    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]})
    builder.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "A"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "B"],
            "target_layer": ["0", "0"],
            "w": [1, 2],
        }
    )
    builder.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["w"],
        drop_na=False,
        drop_duplicates=False,
    )

    d = export_info(core.graph, mode="e", return_type="dict")
    # keys must equal the number of edges
    assert set(d.keys()) == set(range(core.graph.num_edges()))
    # confirm properties
    assert any(info["w"] == 1 for info in d.values())
    assert any(info["w"] == 2 for info in d.values())


# --- 4. Property name filtering --------------------------------------------


def test_export_with_prop_names_subset(simple_graph):
    """
    Specifying prop_names should restrict columns to exactly ['v_int'] + prop_names.
    """
    core = simple_graph
    df = export_info(core.graph, mode="v", prop_names=["score"], return_type="pandas")
    assert list(df.columns) == ["v_int", "score"]


# --- 5. Noisy printing ------------------------------------------------------


def test_export_noisy_prints(capsys, simple_graph):
    """
    noisy=True should print a line per item to stdout.
    """
    core = simple_graph
    _ = export_info(core.graph, mode="v", noisy=True, return_type="list")
    out = capsys.readouterr().out
    assert "Vertex 0:" in out and "Vertex 1:" in out


# --- 6. GraphView round-trip ------------------------------------------------


def test_export_on_graphview(chain_graph_and_searcher):
    """
    Exporting from a filtered GraphView returns only those vertices/edges in the view.
    """
    core, searcher = chain_graph_and_searcher
    # take only B→C by viewing downstream-1 from B
    gv = searcher.search(start_node_idx=1, max_dist=1, direction="downstream", show_plot=False)
    df_full = export_info(core.graph, mode="v", return_type="pandas")
    df_view = export_info(gv, mode="v", return_type="pandas")
    assert len(df_view) < len(df_full)
    assert set(df_view.v_int).issubset(set(df_full.v_int))


# --- 7. Invalid mode / return_type errors ----------------------------------


def test_export_invalid_mode(simple_graph):
    """
    Passing an unsupported mode raises ValueError.
    """
    with pytest.raises(ValueError):
        export_info(simple_graph.graph, mode="x")


def test_export_invalid_return_type(simple_graph):
    """
    Passing an unsupported return_type raises ValueError.
    """
    with pytest.raises(ValueError):
        export_info(simple_graph.graph, mode="v", return_type="xml")


def test_export_vertices_to_dict_and_unknown_prop_raises(simple_graph):
    core = simple_graph
    d = export_info(core.graph, mode="v", return_type="dict")
    assert isinstance(d, dict) and set(d.keys()) == set(range(core.graph.num_vertices()))
    # entries include at least layer/node hashes and sample prop
    any_entry = next(iter(d.values()))
    assert "layer_hash" in any_entry and "node_id_hash" in any_entry
    with pytest.raises(ValueError):
        export_info(core.graph, mode="v", prop_names=["not_present"], return_type="pandas")


# --- 8. Edge IDs when no edge_index ----------------------------------------


def test_export_edges_e_id_none(builder_and_core):
    """
    If the graph has no 'edge_index', exported 'e_id' will be the only type.
    """
    builder, core = builder_and_core
    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]})
    builder.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    # add one edge without any index
    builder.add_edges_from_dataframe(
        pd.DataFrame(
            {"source_id": ["A"], "source_layer": ["0"], "target_id": ["B"], "target_layer": ["0"]}
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )

    lst = export_info(core.graph, mode="e", return_type="list")
    assert all(isinstance(d["e_id"], int) for d in lst)


def _build_ab_edge():
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    b.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]}),
        "node_id",
        "layer",
        drop_na=False,
    )
    b.add_edges_from_dataframe(
        pd.DataFrame(
            {"source_id": ["A"], "source_layer": ["0"], "target_id": ["B"], "target_layer": ["0"]}
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    return core


def test_export_edges_property_name_collision_excluded():
    """
    If an edge property collides with built-in columns ('source','target','e_id'),
    the exporter must NOT overwrite them. We simulate a collision by manually
    creating an edge prop named 'source'. The DataFrame's 'source' column must
    remain the integer vertex index from the graph, and there must be no second
    conflicting column exported.

    Old exporter would overwrite 'source' with the property value (FAIL).
    """
    core = _build_ab_edge()
    g = core.graph
    # Manually attach a colliding edge property called 'source'
    ep_src = g.new_edge_property("string")
    for e in g.edges():
        ep_src[e] = "BROKEN_IF_YOU_SEE_ME"
    g.ep["source"] = ep_src

    df = export_info(g, mode="e", return_type="pandas")

    # Must have exactly these columns (prop 'source' is excluded to avoid collision)
    assert set(df.columns) == {"e_id", "source", "target"}
    # And 'source' must remain an int vertex index (0 for A)
    assert df.loc[df.index[0], "source"] == 0
    assert isinstance(df.loc[df.index[0], "source"], (int,))  # not a string


def test_export_edges_vector_prop_fallback():
    """
    When any requested edge property is non-scalar (e.g., vector<double>),
    exporter should fall back to the item-by-item path and still export values.
    """
    core = _build_ab_edge()
    g = core.graph

    rgba = g.new_edge_property("vector<double>")
    w = g.new_edge_property("int")
    for e in g.edges():
        rgba[e] = [0.1, 0.2, 0.3, 1.0]
        w[e] = 7
    g.ep["rgba"] = rgba
    g.ep["w"] = w

    df = export_info(g, mode="e", prop_names=["w", "rgba"], return_type="pandas")
    assert list(df.columns) == ["e_id", "source", "target", "w", "rgba"]
    assert df["w"].iloc[0] == 7
    # vector<double> comes back as a list-like; check length and contents
    vec = df["rgba"].iloc[0]
    assert isinstance(vec, (list, tuple, np.ndarray))
    assert len(vec) == 4 and abs(vec[0] - 0.1) < 1e-9


def test_export_edges_on_graphview_subset_ids_preserved():
    """
    Exporting from a filtered GraphView should yield only those edges,
    and e_id values should be a subset of the full graph's e_ids.
    """
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    b.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"]}),
        "node_id",
        "layer",
        drop_na=False,
    )
    # A->B, A->C, B->C
    b.add_edges_from_dataframe(
        pd.DataFrame(
            {
                "source_id": ["A", "A", "B"],
                "source_layer": ["0", "0", "0"],
                "target_id": ["B", "C", "C"],
                "target_layer": ["0", "0", "0"],
            }
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    g = core.graph

    # View: keep only edges whose target is C (vertex index 2)
    gv = GraphView(g, efilt=lambda e: int(e.target()) == 2)

    full_df = export_info(g, mode="e", return_type="pandas")
    view_df = export_info(gv, mode="e", return_type="pandas")

    # Edges in the view are a strict subset by e_id
    assert set(view_df["e_id"]).issubset(set(full_df["e_id"]))
    # And they all point to target=2
    assert set(view_df["target"]) == {2}


def test_export_edges_unknown_prop_raises_valueerror():
    """
    Asking for a non-existent edge property should raise ValueError with a clear message.
    Old exporter would leak a KeyError (different type).
    """
    core = _build_ab_edge()
    with pytest.raises(ValueError):
        export_info(core.graph, mode="e", prop_names=["definitely_not_here"], return_type="pandas")


def test_export_edges_collision_names_in_prop_names_are_ignored():
    """
    If the caller explicitly asks for prop_names containing built-in names
    (e.g., 'source','target','e_id'), exporter should ignore them and still
    produce a clean table.
    """
    core = _build_ab_edge()
    # also add a real scalar edge prop to verify it still appears
    w = core.graph.new_edge_property("int")
    for e in core.graph.edges():
        w[e] = 5
    core.graph.ep["w"] = w

    df = export_info(
        core.graph, mode="e", prop_names=["e_id", "source", "target", "w"], return_type="pandas"
    )
    assert list(df.columns) == ["e_id", "source", "target", "w"]
    assert df["w"].iloc[0] == 5
