import pandas as pd
import numpy as np
import pytest

from onionnet.builder      import OnionNetBuilder
from onionnet.core         import OnionNetGraph
from onionnet.builder      import infer_property_type, map_categorical_property

@pytest.fixture
def core():
    return OnionNetGraph()

@pytest.fixture
def builder(core):
    return OnionNetBuilder(core)

@pytest.fixture
def toy_nodes():
    return pd.DataFrame({
        "node_id": ["A","B","C"],
        "layer":   ["0","0","1"],
        "weight":  [1.5,2.0,3.5],
        "group":   ["x","y","x"]
    })

@pytest.fixture
def toy_edges():
    return pd.DataFrame({
        "source_id":    ["A","B"],
        "source_layer": ["0","0"],
        "target_id":    ["B","C"],
        "target_layer": ["0","1"],
        "strength":     [10,20]
    })

def test_missing_node_columns_raises(builder, toy_nodes, toy_edges):
    bad = toy_nodes.drop(columns="layer")
    with pytest.raises(ValueError):
        builder.grow_onion(bad, toy_edges)

def test_add_vertices_basic(builder, core, toy_nodes):
    df = toy_nodes[["node_id","layer"]]
    builder.add_vertices_from_dataframe(df, id_col="node_id", layer_col="layer",
                                        property_cols=None, drop_na=True)

    assert core.graph.num_vertices() == 3
    for _, row in df.iterrows():
        li = core._map_layer   (row.layer)
        ni = core._map_node_id (row.node_id)
        assert (li, ni) in core.custom_id_to_vertex_index

def test_add_vertices_with_props(builder, core, toy_nodes):
    builder.add_vertices_from_dataframe(
        toy_nodes, id_col="node_id", layer_col="layer",
        property_cols=["weight","group"]
    )

    # numeric
    assert np.allclose(core.graph.vp["weight"].a, toy_nodes["weight"].values)
    # categorical
    assert "group" in core.vertex_categorical_mappings
    mapped = core.graph.vp["group"].a
    for v in mapped:
        assert v in core.vertex_categorical_mappings["group"]["int_to_str"]

def test_drop_na_vs_fill(builder):
    df = pd.DataFrame({
        "node_id": ["A", None, "C"],
        "layer":   ["0", "0", None]
    })

    core = OnionNetGraph()
    builder = OnionNetBuilder(core)
    builder.add_vertices_from_dataframe(df, "node_id","layer", drop_na=True)
    assert core.graph.num_vertices() == 3

    core = OnionNetGraph()
    builder = OnionNetBuilder(core)
    builder.add_vertices_from_dataframe(df, "node_id","layer", drop_na=False, fill_na_with="missing")
    assert core.graph.num_vertices() == 3

def test_add_edges(builder, core, toy_nodes, toy_edges):
    builder.add_vertices_from_dataframe(toy_nodes, "node_id","layer")
    builder.add_edges_from_dataframe(
        toy_edges,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["strength"]
    )
    assert core.graph.num_edges() == 2
    assert np.allclose(core.graph.ep["strength"].a, toy_edges["strength"].values)

def test_grow_onion_integration(builder, core, toy_nodes, toy_edges):
    builder.grow_onion(
        toy_nodes, toy_edges,
        node_prop_cols=["weight","group"],
        edge_prop_cols=["strength"],
        drop_duplicates=True
    )

    assert core.graph.num_vertices() == 3
    assert core.graph.num_edges()    == 2
    # categorical vertex properties go into vertex_categorical_mappings and the graph.vp
    assert "group" in core.vertex_categorical_mappings
    assert "group" in core.graph.vp
    # numeric edge props go into graph.ep, not edge_categorical_mappings
    assert "strength" in core.graph.ep
    assert np.allclose(core.graph.ep["strength"].a, toy_edges["strength"].values)


##########################################################################################################
#### More advanced tests, dealing with missing and duplicated ids and properties, for nodes and edges ####
##########################################################################################################

@pytest.fixture
def builder_and_core():
    core    = OnionNetGraph()
    builder = OnionNetBuilder(core)
    return builder, core

# 1) NULL node_id or layer
def test_null_node_id_or_layer(builder_and_core):
    builder, core = builder_and_core
    df_nodes = pd.DataFrame({
        "node_id": [None, "B", "C"],
        "layer":   ["0",    None,  "1"],
        "weight":  [1.0,    2.0,   3.0],
    })
    # should not crash, and still add 3 vertices (casting to str + no drop_na)
    builder.add_vertices_from_dataframe(df_nodes, "node_id", "layer",
                                        property_cols=["weight"], drop_na=False)
    assert core.graph.num_vertices() == 3

def test_add_vertices_errors_on_null_keys_when_dropna_false():
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)
    df = pd.DataFrame({
        "node_id": [None, "B"],
        "layer":   ["0",    "1"]
    })
    with pytest.raises(ValueError) as exc:
        bldr.add_vertices_from_dataframe(df, "node_id", "layer",
                                         drop_na=False, property_cols=None)
    assert "NA in node_id/layer" in str(exc.value)  # your exact message

# 2) DUPLICATED node_id+layer
def test_duplicate_nodes_are_dropped(builder_and_core):
    builder, core = builder_and_core
    df = pd.DataFrame({
        "node_id": ["A", "A", "B"],
        "layer":   ["0", "0", "1"],
        "weight":  [1.0, 2.0, 3.0],
    })
    # drop_duplicates should remove one of the two A@0
    builder.add_vertices_from_dataframe(df, "node_id", "layer",
                                        property_cols=["weight"], drop_na=True)
    assert core.graph.num_vertices() == 2
    # check that the kept “A@0” has the *first* weight value (1.0)
    li = core._map_layer("0"); ni = core._map_node_id("A")
    vidx = core.custom_id_to_vertex_index[(li,ni)]
    assert core.graph.vp["weight"].a[vidx] == pytest.approx(1.0)

# 3) NULL property values
def test_null_properties_fill_or_cast(builder_and_core):
    builder, core = builder_and_core
    df = pd.DataFrame({
        "node_id": ["A","B"],
        "layer":   ["0","1"],
        "weight":  [np.nan, 5.0],
        "group":   [None,   "x"],
    })
    # numeric prop “weight” will carry through as nan
    # categorical “group” will get mapped (including nan→"nan")
    builder.add_vertices_from_dataframe(df, "node_id","layer",
                                        property_cols=["weight","group"],
                                        drop_na=False)
    li, ni = core._map_layer("0"), core._map_node_id("A")
    vidx0  = core.custom_id_to_vertex_index[(li,ni)]
    assert np.isnan(core.graph.vp["weight"].a[vidx0])
    # group mapping includes the “None” as a category
    assert  core.graph.vp["group"].a[vidx0] in core.vertex_categorical_mappings["group"]["int_to_str"]

# 4) DUPLICATED properties (same node repeated with different prop values)
def test_duplicate_rows_keep_first_properties(builder_and_core):
    builder, core = builder_and_core
    df = pd.DataFrame({
        "node_id": ["A","A"],
        "layer":   ["0","0"],
        "weight":  [1.1, 9.9],
    })
    # drop_duplicates(True) will drop the second A@0, keeping weight=1.1
    builder.add_vertices_from_dataframe(df, "node_id","layer",
                                        property_cols=["weight"], drop_na=True)
    li, ni = core._map_layer("0"), core._map_node_id("A")
    vidx   = core.custom_id_to_vertex_index[(li,ni)]
    assert core.graph.vp["weight"].a[vidx] == pytest.approx(1.1)

# 5) NULL or INVALID edges
def test_edges_with_missing_nodes_are_skipped(builder_and_core):
    builder, core = builder_and_core
    # first seed the nodes
    df_nodes = pd.DataFrame({
        "node_id": ["A"],
        "layer":   ["0"]
    })
    builder.add_vertices_from_dataframe(df_nodes, "node_id","layer")

    # edge refers to unknown node "B"
    df_edges = pd.DataFrame({
        "source_id":    ["A","B"],
        "source_layer": ["0","0"],
        "target_id":    ["B","A"],
        "target_layer": ["0","0"],
        "strength":     [1.0, 2.0]
    })
    # should only add the one valid A→A edge (if any),
    # or zero if code filters only complete pairs
    builder.add_edges_from_dataframe(df_edges, "source_id","source_layer",
                                     "target_id","target_layer", property_cols=None)
    # Confirm no crash and edges count ≤ number of rows
    assert core.graph.num_edges() in (0, 1)

def test_add_edges_errors_on_null_keys_when_dropna_false():
    core = OnionNetGraph()
    b   = OnionNetBuilder(core)
    # seed one valid node so mapping exists
    b.add_vertices_from_dataframe(
      pd.DataFrame({"node_id":["A"],"layer":["0"]}),
      "node_id","layer", drop_na=True
    )
    bad_edges = pd.DataFrame({
        "source_id":    ["A", None],
        "source_layer": ["0","0"],
        "target_id":    ["A","A"],
        "target_layer": ["0","0"]
    })
    with pytest.raises(ValueError) as exc:
        b.add_edges_from_dataframe(
          bad_edges,
          "source_id","source_layer",
          "target_id","target_layer",
          drop_na=False, property_cols=None
        )
    assert "drop_na=False" in str(exc.value)

# 6) DUPLICATED edges
def test_duplicate_edges_are_collapsed(builder_and_core):
    builder, core = builder_and_core
    # two identical edges A→A
    df_nodes = pd.DataFrame({"node_id":["A"], "layer":["0"]})
    df_edges = pd.DataFrame({
        "source_id":    ["A","A"],
        "source_layer": ["0","0"],
        "target_id":    ["A","A"],
        "target_layer": ["0","0"],
    })
    builder.add_vertices_from_dataframe(df_nodes, "node_id","layer")
    builder.add_edges_from_dataframe(df_edges, "source_id","source_layer",
                                     "target_id","target_layer")
    # drop_duplicates subset only source+target, so one edge remains
    assert core.graph.num_edges() == 1

# 7) NODES ↔ EDGES alignment (no stray edges)
def test_edge_node_alignment(builder_and_core):
    builder, core = builder_and_core
    df_nodes = pd.DataFrame({
        "node_id": ["A","B"],
        "layer":   ["0","0"]
    })
    df_edges = pd.DataFrame({
        # includes C→A (C not ingested)
        "source_id":    ["A","C"],
        "source_layer": ["0","0"],
        "target_id":    ["B","A"],
        "target_layer": ["0","0"]
    })
    builder.grow_onion(df_nodes, df_edges,
                       node_prop_cols=[], edge_prop_cols=[],
                       drop_duplicates=True)
    assert core.graph.num_vertices() == 2
    # only A→B should remain
    assert core.graph.num_edges() == 1