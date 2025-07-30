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


# 12) Empty inputs should produce no vertices/edges but no crash
def test_empty_inputs(builder_and_core):
    builder, core = builder_and_core
    empty_nodes = pd.DataFrame(columns=["node_id","layer"])
    empty_edges = pd.DataFrame(columns=["source_id","source_layer","target_id","target_layer"])
    builder.grow_onion(empty_nodes, empty_edges,
                       node_prop_cols=[], edge_prop_cols=[],
                       drop_na=False, drop_duplicates=True)
    assert core.graph.num_vertices() == 0
    assert core.graph.num_edges()    == 0

# 13) Self-loops supported and duplicates kept by default in add_edges
def test_self_loops_are_supported(builder_and_core, toy_nodes):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(toy_nodes, "node_id","layer", drop_na=False)
    self_loops = pd.DataFrame({
        "source_id":    ["A","B","A"],
        "source_layer": ["0","0","0"],
        "target_id":    ["A","B","A"],
        "target_layer": ["0","0","0"],
        "strength":     [1,2,1]
    })
    builder.add_edges_from_dataframe(self_loops,
                                      source_id_col="source_id", source_layer_col="source_layer",
                                      target_id_col="target_id", target_layer_col="target_layer",
                                      property_cols=None, drop_na=False)
    # three edges (including duplicate A->A) should be present
    assert core.graph.num_edges() == 3

# 14) Mixed-type IDs and layers are cast to str
def test_mixed_type_ids_and_layers_cast_to_str(builder_and_core):
    builder, core = builder_and_core
    df = pd.DataFrame({
        "node_id": [1, 2, 3],
        "layer":   [0,  1,  1]
    })
    builder.add_vertices_from_dataframe(df, "node_id","layer", drop_na=False)
    assert core.graph.num_vertices() == 3
    for raw_id, raw_layer in zip(df.node_id, df.layer):
        li = core._map_layer(str(raw_layer))
        ni = core._map_node_id(str(raw_id))
        assert (li, ni) in core.custom_id_to_vertex_index

# 15) string_override=True forces all props to categorical
def test_string_override_forces_categorical(builder_and_core, toy_nodes):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(toy_nodes, "node_id","layer",
                                        property_cols=["weight"], drop_na=False,
                                        string_override=True)
    assert "weight" in core.vertex_categorical_mappings
    vt = core.graph.vp["weight"].value_type()
    assert vt.startswith("int")

# 16) custom property_types override inference
def test_custom_property_types_override(builder_and_core, toy_nodes, toy_edges):
    builder, core = builder_and_core
    # seed A and B
    builder.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]}),
        "node_id","layer", drop_na=False)
    # treat strength as categorical despite numeric
    one_edge = pd.DataFrame({
        "source_id":["A"], "source_layer":["0"],
        "target_id":["B"], "target_layer":["0"],
        "strength":[42]
    })
    builder.add_edges_from_dataframe(
        one_edge,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["strength"], drop_na=False,
        property_types={"strength":"str"}
    )
    assert "strength" in core.edge_categorical_mappings

# 17) nonexistent prop column raises ValueError
def test_nonexistent_prop_column_raises(builder_and_core, toy_nodes):
    builder, core = builder_and_core
    with pytest.raises(ValueError):
        builder.add_vertices_from_dataframe(
            toy_nodes, "node_id","layer",
            property_cols=["not_a_column"], drop_na=False
        )

# 18) extreme values and empty strings handled correctly
def test_extreme_and_empty_strings(builder_and_core):
    builder, core = builder_and_core
    df = pd.DataFrame({
        "node_id": ["","   ","X"],
        "layer":   ["0","1","1"],
        "weight":  [np.nan, np.inf, -np.inf],
    })
    builder.add_vertices_from_dataframe(df, "node_id","layer",
                                        property_cols=["weight"], drop_na=False)
    assert core.graph.num_vertices() == 3
    w = core.graph.vp["weight"].a
    assert np.isnan(w[0]) and np.isposinf(w[1]) and np.isneginf(w[2])

# 19) large-scale (smoke) ingest
@pytest.mark.slow
def test_bulk_ingest_performance(builder_and_core):
    builder, core = builder_and_core
    N=10_000
    nodes = pd.DataFrame({
        "node_id": np.arange(N).astype(str),
        "layer":   np.zeros(N,dtype=int).astype(str),
    })
    edges = pd.DataFrame({
        "source_id":    np.random.choice(nodes.node_id, size=2*N),
        "source_layer": ["0"]*(2*N),
        "target_id":    np.random.choice(nodes.node_id, size=2*N),
        "target_layer": ["0"]*(2*N),
    })
    builder.grow_onion(nodes, edges,
                       node_prop_cols=[], edge_prop_cols=[],
                       drop_na=False, drop_duplicates=True)
    assert core.graph.num_vertices() == N
    assert core.graph.num_edges() <= 2*N
