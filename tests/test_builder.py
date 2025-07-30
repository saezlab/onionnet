import pandas as pd
import numpy as np
import pytest

from onionnet.builder import OnionNetBuilder
from onionnet.core    import OnionNetGraph

@pytest.fixture
def core():
    return OnionNetGraph()

@pytest.fixture
def builder(core):
    return OnionNetBuilder(core)

@pytest.fixture
def builder_and_core():
    core = OnionNetGraph()
    builder = OnionNetBuilder(core)
    return builder, core

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

# 1) Missing required columns in grow_onion
def test_missing_node_columns_raises(builder, toy_nodes, toy_edges):
    bad = toy_nodes.drop(columns="layer")
    with pytest.raises(ValueError):
        builder.grow_onion(bad, toy_edges)

# 2) Basic add_vertices: include all rows with drop_na=False
def test_add_vertices_basic(builder, core, toy_nodes):
    df = toy_nodes[["node_id","layer"]]
    builder.add_vertices_from_dataframe(df, id_col="node_id", layer_col="layer",
                                        property_cols=None, drop_na=False)
    assert core.graph.num_vertices() == 3
    for _, row in df.iterrows():
        li = core._map_layer(row.layer)
        ni = core._map_node_id(row.node_id)
        assert (li, ni) in core.custom_id_to_vertex_index

# 3) add_vertices drops missing when drop_na=True
def test_add_vertices_drop_na_true(builder):
    core = OnionNetGraph()
    bu = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id":["A", None, "C"], "layer":["0","0",None]})
    bu.add_vertices_from_dataframe(df, "node_id","layer", property_cols=None, drop_na=True)
    assert core.graph.num_vertices() == 1

# 4) add_vertices raises on missing when drop_na=False
def test_add_vertices_drop_na_false_raises(builder):
    core = OnionNetGraph(); bu = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id":["A", None], "layer":["0","1"]})
    with pytest.raises(ValueError) as exc:
        bu.add_vertices_from_dataframe(df, "node_id","layer", property_cols=None, drop_na=False)
    assert "drop_na=False" in str(exc.value)

# 5) add_vertices with props
def test_add_vertices_with_props(builder, core, toy_nodes):
    builder.add_vertices_from_dataframe(toy_nodes, "node_id","layer",
                                        property_cols=["weight","group"], drop_na=False)
    assert np.allclose(core.graph.vp["weight"].a, toy_nodes["weight"].values)
    assert "group" in core.vertex_categorical_mappings
    for val in core.graph.vp["group"].a:
        assert val in core.vertex_categorical_mappings["group"]["int_to_str"]

# 6) Duplicate nodes removed via grow_onion(drop_duplicates=True)
def test_duplicate_nodes_via_grow(builder_and_core, toy_nodes, toy_edges):
    builder, core = builder_and_core
    dup = pd.concat([toy_nodes, toy_nodes.iloc[[0]]], ignore_index=True)
    builder.grow_onion(dup, toy_edges,
                       node_prop_cols=["weight"], edge_prop_cols=[],
                       drop_na=False, drop_duplicates=True)
    assert core.graph.num_vertices() == 3

# 7) Basic add_edges: include all rows with drop_na=False
def test_add_edges_basic(builder, core, toy_nodes, toy_edges):
    builder.add_vertices_from_dataframe(toy_nodes, "node_id","layer", drop_na=False)
    builder.add_edges_from_dataframe(toy_edges, "source_id","source_layer",
                                      "target_id","target_layer",
                                      property_cols=["strength"], drop_na=False)
    assert core.graph.num_edges() == 2
    assert np.allclose(core.graph.ep["strength"].a, toy_edges["strength"].values)

# 8) add_edges drops missing when drop_na=True
def test_add_edges_drop_na_true(builder_and_core):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(pd.DataFrame({"node_id":["A"],"layer":["0"]}),
                                        "node_id","layer", drop_na=False)
    df_e = pd.DataFrame({"source_id":["A",None],"source_layer":["0","0"],
                         "target_id":["A","A"],"target_layer":["0","0"],
                         "strength":[5,6]})
    builder.add_edges_from_dataframe(df_e, "source_id","source_layer",
                                      "target_id","target_layer",
                                      property_cols=["strength"], drop_na=True)
    assert core.graph.num_edges() == 1

# 9) add_edges raises on missing when drop_na=False
def test_add_edges_drop_na_false_raises(builder_and_core):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(pd.DataFrame({"node_id":["A"],"layer":["0"]}),
                                        "node_id","layer", drop_na=False)
    df = pd.DataFrame({"source_id":["A",None],"source_layer":["0","0"],
                       "target_id":["A","A"],"target_layer":["0","0"]})
    with pytest.raises(ValueError) as exc:
        builder.add_edges_from_dataframe(df, "source_id","source_layer",
                                          "target_id","target_layer",
                                          property_cols=None, drop_na=False)
    assert "drop_na=False" in str(exc.value)

# 10) Duplicate edges removed via grow_onion(drop_duplicates=True)
def test_duplicate_edges_via_grow(builder_and_core, toy_nodes, toy_edges):
    builder, core = builder_and_core
    dup_e = pd.concat([toy_edges, toy_edges.iloc[[0]]], ignore_index=True)
    builder.grow_onion(toy_nodes, dup_e,
                       node_prop_cols=[], edge_prop_cols=["strength"],
                       drop_na=False, drop_duplicates=True)
    assert core.graph.num_edges() == 2

# 11) Edge-node alignment via grow_onion
def test_edge_node_alignment(builder, core):
    df_n = pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]})
    df_e = pd.DataFrame({"source_id":["A","C"],"source_layer":["0","0"],
                         "target_id":["B","A"],"target_layer":["0","0"]})
    builder.grow_onion(df_n, df_e,
                       node_prop_cols=[], edge_prop_cols=[],
                       drop_na=False, drop_duplicates=True)
    assert core.graph.num_vertices() == 2
    assert core.graph.num_edges() == 1