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
    df = pd.DataFrame({
        "node_id": ["A","B"],
        "layer":   ["0","0"],
        "score":   [10,20]
    })
    b.add_vertices_from_dataframe(df, "node_id","layer", property_cols=["score"], drop_na=False)
    # connect A->B so that edges exist
    b.add_edges_from_dataframe(
        pd.DataFrame({
            "source_id":    ["A"],
            "source_layer": ["0"],
            "target_id":    ["B"],
            "target_layer": ["0"],
        }),
        "source_id","source_layer","target_id","target_layer",
        property_cols=None, drop_na=False
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
    df_nodes = pd.DataFrame({
        "node_id": ["A","B","C"],
        "layer":   ["0","0","0"]
    })
    b.add_vertices_from_dataframe(df_nodes, "node_id","layer", drop_na=False)
    # edges A→B, B→C
    df_edges = pd.DataFrame({
        "source_id":    ["A","B"],
        "source_layer": ["0","0"],
        "target_id":    ["B","C"],
        "target_layer": ["0","0"],
    })
    b.add_edges_from_dataframe(
        df_edges,
        "source_id","source_layer","target_id","target_layer",
        property_cols=None, drop_na=False
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
    df = pd.DataFrame({
        "node_id": ["A","B","C"],
        "layer":   ["0","0","0"],
        "grp":     ["x","y","x"]
    })
    builder.add_vertices_from_dataframe(df, "node_id","layer", property_cols=["grp"], drop_na=False)

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
    df_n = pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]})
    builder.add_vertices_from_dataframe(df_n, "node_id","layer", drop_na=False)
    df_e = pd.DataFrame({
        "source_id":    ["A"],
        "source_layer": ["0"],
        "target_id":    ["B"],
        "target_layer": ["0"],
        "strength":     [42]
    })
    builder.add_edges_from_dataframe(
        df_e, "source_id","source_layer","target_id","target_layer",
        property_cols=["strength"], drop_na=False
    )

    lst = export_info(core.graph, mode="e", return_type="list")
    # should be a list with one dict
    assert isinstance(lst, list) and len(lst) == 1
    entry = lst[0]
    assert set(entry.keys()) == {"e_id","source","target","strength"}
    assert entry["strength"] == 42
    assert entry["source"] == 0 and entry["target"] == 1


# --- 3. Edge export to dict keyed by e_id ----------------------------------

def test_export_edges_to_dict(builder_and_core):
    """
    return_type='dict' should key the output by edge ID.
    """
    builder, core = builder_and_core
    # seed two nodes + two edges
    df_n = pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]})
    builder.add_vertices_from_dataframe(df_n, "node_id","layer", drop_na=False)
    df_e = pd.DataFrame({
        "source_id":    ["A","A"],
        "source_layer": ["0","0"],
        "target_id":    ["B","B"],
        "target_layer": ["0","0"],
        "w":            [1,2]
    })
    builder.add_edges_from_dataframe(
        df_e,"source_id","source_layer","target_id","target_layer",
        property_cols=["w"], drop_na=False, drop_duplicates=False
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
    assert list(df.columns) == ["v_int","score"]


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
    gv = searcher.search(start_node_idx=1, max_dist=1, direction='downstream', show_plot=False)
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


# --- 8. Edge IDs when no edge_index ----------------------------------------

def test_export_edges_e_id_none(builder_and_core):
    """
    If the graph has no 'edge_index', exported 'e_id' will be the only type.
    """
    builder, core = builder_and_core
    df_n = pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]})
    builder.add_vertices_from_dataframe(df_n, "node_id","layer", drop_na=False)
    # add one edge without any index
    builder.add_edges_from_dataframe(
        pd.DataFrame({
            "source_id":["A"],"source_layer":["0"],
            "target_id":["B"],"target_layer":["0"]
        }),
        "source_id","source_layer","target_id","target_layer",
        property_cols=None, drop_na=False
    )

    lst = export_info(core.graph, mode="e", return_type="list")
    assert all(isinstance(d["e_id"], int) for d in lst)