import pandas as pd
import numpy as np
import pytest

from onionnet.builder import OnionNetBuilder
from onionnet.core import OnionNetGraph


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
    return pd.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "layer": ["0", "0", "1"],
            "weight": [1.5, 2.0, 3.5],
            "group": ["x", "y", "x"],
        }
    )


@pytest.fixture
def toy_edges():
    return pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "C"],
            "target_layer": ["0", "1"],
            "strength": [10, 20],
        }
    )


# 1) Missing required columns in grow_onion
def test_missing_node_columns_raises(builder, toy_nodes, toy_edges):
    bad = toy_nodes.drop(columns="layer")
    with pytest.raises(ValueError):
        builder.grow_onion(bad, toy_edges)


# 2) Basic add_vertices: include all rows with drop_na=False
def test_add_vertices_basic(builder, core, toy_nodes):
    df = toy_nodes[["node_id", "layer"]]
    builder.add_vertices_from_dataframe(
        df, id_col="node_id", layer_col="layer", property_cols=None, drop_na=False
    )
    assert core.graph.num_vertices() == 3
    for _, row in df.iterrows():
        li = core._map_layer(row.layer)
        ni = core._map_node_id(row.node_id)
        assert (li, ni) in core.custom_id_to_vertex_index


# 3) add_vertices drops missing when drop_na=True
def test_add_vertices_drop_na_true(builder):
    core = OnionNetGraph()
    bu = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A", None, "C"], "layer": ["0", "0", None]})
    bu.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=None, drop_na=True)
    assert core.graph.num_vertices() == 1


# 4) add_vertices raises on missing when drop_na=False
def test_add_vertices_drop_na_false_raises(builder):
    core = OnionNetGraph()
    bu = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A", None], "layer": ["0", "1"]})
    with pytest.raises(ValueError) as exc:
        bu.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=None, drop_na=False)
    assert "drop_na=False" in str(exc.value)


# 5) add_vertices with props
def test_add_vertices_with_props(builder, core, toy_nodes):
    builder.add_vertices_from_dataframe(
        toy_nodes, "node_id", "layer", property_cols=["weight", "group"], drop_na=False
    )
    assert np.allclose(core.graph.vp["weight"].a, toy_nodes["weight"].values)
    assert "group" in core.vertex_categorical_mappings
    for val in core.graph.vp["group"].a:
        assert val in core.vertex_categorical_mappings["group"]["int_to_str"]


# 6) Duplicate nodes removed via grow_onion(drop_duplicates=True)
def test_duplicate_nodes_via_grow(builder_and_core, toy_nodes, toy_edges):
    builder, core = builder_and_core
    dup = pd.concat([toy_nodes, toy_nodes.iloc[[0]]], ignore_index=True)
    builder.grow_onion(
        dup,
        toy_edges,
        node_prop_cols=["weight"],
        edge_prop_cols=[],
        drop_na=False,
        drop_duplicates=True,
    )
    assert core.graph.num_vertices() == 3


# 7) Basic add_edges: include all rows with drop_na=False
def test_add_edges_basic(builder, core, toy_nodes, toy_edges):
    builder.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    builder.add_edges_from_dataframe(
        toy_edges,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["strength"],
        drop_na=False,
    )
    assert core.graph.num_edges() == 2
    assert np.allclose(core.graph.ep["strength"].a, toy_edges["strength"].values)


# 8) add_edges drops missing when drop_na=True
def test_add_edges_drop_na_true(builder_and_core):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A"], "layer": ["0"]}), "node_id", "layer", drop_na=False
    )
    df_e = pd.DataFrame(
        {
            "source_id": ["A", None],
            "source_layer": ["0", "0"],
            "target_id": ["A", "A"],
            "target_layer": ["0", "0"],
            "strength": [5, 6],
        }
    )
    builder.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["strength"],
        drop_na=True,
    )
    assert core.graph.num_edges() == 1


# 9) add_edges raises on missing when drop_na=False
def test_add_edges_drop_na_false_raises(builder_and_core):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A"], "layer": ["0"]}), "node_id", "layer", drop_na=False
    )
    df = pd.DataFrame(
        {
            "source_id": ["A", None],
            "source_layer": ["0", "0"],
            "target_id": ["A", "A"],
            "target_layer": ["0", "0"],
        }
    )
    with pytest.raises(ValueError) as exc:
        builder.add_edges_from_dataframe(
            df,
            "source_id",
            "source_layer",
            "target_id",
            "target_layer",
            property_cols=None,
            drop_na=False,
        )
    assert "drop_na=False" in str(exc.value)


# 10) Duplicate edges removed via grow_onion(drop_duplicates=True)
def test_duplicate_edges_via_grow(builder_and_core, toy_nodes, toy_edges):
    builder, core = builder_and_core
    dup_e = pd.concat([toy_edges, toy_edges.iloc[[0]]], ignore_index=True)
    builder.grow_onion(
        toy_nodes,
        dup_e,
        node_prop_cols=[],
        edge_prop_cols=["strength"],
        drop_na=False,
        drop_duplicates=True,
    )
    assert core.graph.num_edges() == 2


# 11) Edge-node alignment via grow_onion
def test_edge_node_alignment(builder, core):
    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]})
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "C"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "A"],
            "target_layer": ["0", "0"],
        }
    )
    builder.grow_onion(
        df_n, df_e, node_prop_cols=[], edge_prop_cols=[], drop_na=False, drop_duplicates=True
    )
    assert core.graph.num_vertices() == 2
    assert core.graph.num_edges() == 1


# 12) Empty inputs should produce no vertices/edges but no crash
def test_empty_inputs(builder_and_core):
    builder, core = builder_and_core
    empty_nodes = pd.DataFrame(columns=["node_id", "layer"])
    empty_edges = pd.DataFrame(columns=["source_id", "source_layer", "target_id", "target_layer"])
    builder.grow_onion(
        empty_nodes,
        empty_edges,
        node_prop_cols=[],
        edge_prop_cols=[],
        drop_na=False,
        drop_duplicates=True,
    )
    assert core.graph.num_vertices() == 0
    assert core.graph.num_edges() == 0


# 13) Self-loops supported (A->A) but duplicates not kept by default in add_edges
def test_self_loops_are_supported(builder_and_core, toy_nodes):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    self_loops = pd.DataFrame(
        {
            "source_id": ["A", "B", "A"],
            "source_layer": ["0", "0", "0"],
            "target_id": ["A", "B", "A"],
            "target_layer": ["0", "0", "0"],
            "strength": [1, 2, 1],
        }
    )
    builder.add_edges_from_dataframe(
        self_loops,
        source_id_col="source_id",
        source_layer_col="source_layer",
        target_id_col="target_id",
        target_layer_col="target_layer",
        property_cols=None,
        drop_na=False,
        drop_duplicates=True,
    )
    # two edges (excluding duplicate A->A) should be present
    assert core.graph.num_edges() == 2


# 14) Mixed-type IDs and layers are cast to str
def test_mixed_type_ids_and_layers_cast_to_str(builder_and_core):
    builder, core = builder_and_core
    df = pd.DataFrame({"node_id": [1, 2, 3], "layer": [0, 1, 1]})
    builder.add_vertices_from_dataframe(df, "node_id", "layer", drop_na=False)
    assert core.graph.num_vertices() == 3
    for raw_id, raw_layer in zip(df.node_id, df.layer):
        li = core._map_layer(str(raw_layer))
        ni = core._map_node_id(str(raw_id))
        assert (li, ni) in core.custom_id_to_vertex_index


# 15) string_override=True forces all props to categorical
def test_string_override_forces_categorical(builder_and_core, toy_nodes):
    builder, core = builder_and_core
    builder.add_vertices_from_dataframe(
        toy_nodes, "node_id", "layer", property_cols=["weight"], drop_na=False, string_override=True
    )
    assert "weight" in core.vertex_categorical_mappings
    vt = core.graph.vp["weight"].value_type()
    assert vt.startswith("int")


# 16) custom property_types override inference
def test_custom_property_types_override(builder_and_core, toy_nodes, toy_edges):
    builder, core = builder_and_core
    # seed A and B
    builder.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]}),
        "node_id",
        "layer",
        drop_na=False,
    )
    # treat strength as categorical despite numeric
    one_edge = pd.DataFrame(
        {
            "source_id": ["A"],
            "source_layer": ["0"],
            "target_id": ["B"],
            "target_layer": ["0"],
            "strength": [42],
        }
    )
    builder.add_edges_from_dataframe(
        one_edge,
        source_id_col="source_id",
        source_layer_col="source_layer",
        target_id_col="target_id",
        target_layer_col="target_layer",
        property_cols=["strength"],
        drop_na=False,
        property_types={"strength": "str"},
    )
    assert "strength" in core.edge_categorical_mappings


# 17) nonexistent prop column raises ValueError
def test_nonexistent_prop_column_raises(builder_and_core, toy_nodes):
    builder, core = builder_and_core
    with pytest.raises(ValueError):
        builder.add_vertices_from_dataframe(
            toy_nodes, "node_id", "layer", property_cols=["not_a_column"], drop_na=False
        )


# 18) extreme values and empty strings handled correctly
def test_extreme_and_empty_strings(builder_and_core):
    builder, core = builder_and_core
    df = pd.DataFrame(
        {
            "node_id": ["", "   ", "X"],
            "layer": ["0", "1", "1"],
            "weight": [np.nan, np.inf, -np.inf],
        }
    )
    builder.add_vertices_from_dataframe(
        df, "node_id", "layer", property_cols=["weight"], drop_na=False
    )
    assert core.graph.num_vertices() == 3
    w = core.graph.vp["weight"].a
    assert np.isnan(w[0]) and np.isposinf(w[1]) and np.isneginf(w[2])


# 19) large-scale (smoke) ingest
@pytest.mark.slow
def test_bulk_ingest_performance(builder_and_core):
    builder, core = builder_and_core
    N = 10_000
    nodes = pd.DataFrame(
        {
            "node_id": np.arange(N).astype(str),
            "layer": np.zeros(N, dtype=int).astype(str),
        }
    )
    edges = pd.DataFrame(
        {
            "source_id": np.random.choice(nodes.node_id, size=2 * N),
            "source_layer": ["0"] * (2 * N),
            "target_id": np.random.choice(nodes.node_id, size=2 * N),
            "target_layer": ["0"] * (2 * N),
        }
    )
    builder.grow_onion(
        nodes, edges, node_prop_cols=[], edge_prop_cols=[], drop_na=False, drop_duplicates=True
    )
    assert core.graph.num_vertices() == N
    assert core.graph.num_edges() <= 2 * N


#### Integration Tests ####


def test_multi_layer_cross_edges(builder_and_core):
    """
    Nodes live on layers 0,1,2,1, and edges jump between them.
    The one phantom edge (X→A) should be dropped; the other 4 survive.
    """
    bldr, core = builder_and_core
    nodes = pd.DataFrame(
        {
            "node_id": ["A", "B", "C", "D"],
            "layer": ["0", "1", "2", "1"],
        }
    )
    edges = pd.DataFrame(
        {
            "source_id": ["A", "B", "C", "D", "X"],
            "source_layer": ["0", "1", "2", "1", "0"],
            "target_id": ["B", "C", "D", "A", "A"],
            "target_layer": ["1", "2", "1", "0", "0"],
        }
    )
    bldr.grow_onion(
        nodes, edges, node_prop_cols=[], edge_prop_cols=[], drop_na=False, drop_duplicates=True
    )
    # four real nodes, four valid edges
    assert core.graph.num_vertices() == 4
    assert core.graph.num_edges() == 4

    # Every edge's endpoints map back into our custom_id/bookkeeping
    for e in core.graph.edges():
        s, t = e.source(), e.target()
        assert s in core.vertex_index_to_custom_id
        assert t in core.vertex_index_to_custom_id


def test_incremental_builds_preserve_indices(builder_and_core):
    """
    First ingest A→B, then later add C→A. Ensure A,B keep same indices
    and C is appended—but no duplicates.
    """
    bldr, core = builder_and_core

    # first batch
    n1 = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]})
    e1 = pd.DataFrame(
        {"source_id": ["A"], "source_layer": ["0"], "target_id": ["B"], "target_layer": ["0"]}
    )
    bldr.grow_onion(
        n1, e1, node_prop_cols=[], edge_prop_cols=[], drop_na=False, drop_duplicates=True
    )
    idxA = core.custom_id_to_vertex_index[(core._map_layer("0"), core._map_node_id("A"))]
    idxB = core.custom_id_to_vertex_index[(core._map_layer("0"), core._map_node_id("B"))]
    assert core.graph.num_vertices() == 2
    assert core.graph.num_edges() == 1

    # second batch
    n2 = pd.DataFrame({"node_id": ["C"], "layer": ["1"]})
    e2 = pd.DataFrame(
        {"source_id": ["C"], "source_layer": ["1"], "target_id": ["A"], "target_layer": ["0"]}
    )
    bldr.grow_onion(
        n2, e2, node_prop_cols=[], edge_prop_cols=[], drop_na=False, drop_duplicates=True
    )
    # indices A and B unchanged; C is new
    assert core.custom_id_to_vertex_index[(core._map_layer("0"), core._map_node_id("A"))] == idxA
    assert core.custom_id_to_vertex_index[(core._map_layer("0"), core._map_node_id("B"))] == idxB
    assert core.graph.num_vertices() == 3
    assert core.graph.num_edges() == 2


def test_interleaved_duplicates_and_missing(builder_and_core):
    """
    A DataFrame with both duplicates and some missing keys;
    with drop_na + drop_duplicates, only unique, complete rows survive.
    """
    bldr, core = builder_and_core
    df = pd.DataFrame({"node_id": ["A", "A", None, "C", "C"], "layer": ["0", "0", "1", None, "2"]})
    bldr.add_vertices_from_dataframe(
        df, "node_id", "layer", property_cols=None, drop_na=True
    )  # must drop the None layer row
    # only A@0 and C@2 remain → 2 vertices
    assert core.graph.num_vertices() == 2
    kept = set(core.vertex_index_to_custom_id.values())
    assert kept == {
        (core._map_layer("0"), core._map_node_id("A")),
        (core._map_layer("2"), core._map_node_id("C")),
    }


def test_id_layer_namespacing(builder_and_core):
    """
    Two nodes both named "A" but on different layers must remain distinct.
    """
    bldr, core = builder_and_core
    nodes = pd.DataFrame({"node_id": ["A", "A"], "layer": ["0", "1"]})
    bldr.add_vertices_from_dataframe(nodes, "node_id", "layer", drop_na=False)
    # Should see two distinct entries in custom_id_to_vertex_index
    keys = set(core.custom_id_to_vertex_index.keys())
    assert len(keys) == 2
    assert (core._map_layer("0"), core._map_node_id("A")) in keys
    assert (core._map_layer("1"), core._map_node_id("A")) in keys


def test_repeated_growth_preserves_properties(builder_and_core, toy_nodes, toy_edges):
    """
    Grow once with 'weight', then again with a new 'group' prop.
    You should end up with both properties attached.
    """
    bldr, core = builder_and_core
    # first ingest only weight
    bldr.grow_onion(
        toy_nodes,
        toy_edges,
        node_prop_cols=["weight"],
        edge_prop_cols=[],
        drop_na=False,
        drop_duplicates=False,
    )
    # then re-grow adding 'group'
    # (edges are the same, so drop_duplicates=False to just append new props)
    bldr.grow_onion(
        toy_nodes,
        toy_edges,
        node_prop_cols=["group"],
        edge_prop_cols=[],
        drop_na=False,
        drop_duplicates=False,
    )
    assert "weight" in core.graph.vp
    assert "group" in core.graph.vp
    # vertex count doubled, but both props present on all vertices
    assert core.graph.num_vertices() == len(toy_nodes) * 2


@pytest.mark.slow
def test_bulk_ingest_random_noise(builder_and_core):
    """
    Stress test: 1000 random nodes/edges with some missing & dupes.
    Ensures no edge endpoint ever falls outside the vertex map.
    """
    bldr, core = builder_and_core
    rng = np.random.default_rng(1234)

    # generate 500 nodes, string IDs like 'N42', layers '0','1','2'
    ids = [f"N{i}" for i in rng.integers(0, 100, 500)]
    lays = rng.choice(["0", "1", "2"], size=500)
    df_n = pd.DataFrame({"node_id": ids, "layer": lays})

    # generate 1000 edges, possibly referring to some missing id/layer combos
    s_ids = rng.choice(ids + ["XX", "YY"], size=1000)
    s_lays = rng.choice(["0", "1", "2"], size=1000)
    t_ids = rng.choice(ids + ["ZZ"], size=1000)
    t_lays = rng.choice(["0", "1", "2"], size=1000)
    df_e = pd.DataFrame(
        {"source_id": s_ids, "source_layer": s_lays, "target_id": t_ids, "target_layer": t_lays}
    )

    bldr.grow_onion(
        df_n, df_e, node_prop_cols=[], edge_prop_cols=[], drop_na=False, drop_duplicates=True
    )

    # no more edges than raw rows
    assert core.graph.num_edges() <= len(df_e)

    # every edge endpoint is a known vertex
    for e in core.graph.edges():
        assert e.source() in core.vertex_index_to_custom_id
        assert e.target() in core.vertex_index_to_custom_id


# 20) Missing vertex properties (NaNs) should end up as NaN in vp and still map categoricals consistently.
def test_vertex_props_with_nans(builder_and_core):
    bldr, core = builder_and_core
    df = pd.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "layer": ["0", "0", "0"],
            "p1": [1.0, np.nan, 3.0],
            "p2": ["x", None, "y"],
        }
    )
    bldr.add_vertices_from_dataframe(
        df, "node_id", "layer", property_cols=["p1", "p2"], drop_na=False
    )
    # numeric nan preserved
    arr1 = core.graph.vp["p1"].a
    assert np.isnan(arr1[1])
    # categorical None got its own code
    codes = core.graph.vp["p2"].a
    mapping = core.vertex_categorical_mappings["p2"]["int_to_str"]
    assert mapping[codes[1]] is None
    # original non-null map correctly
    assert mapping[codes[0]] == "x"
    assert mapping[codes[2]] == "y"


# 21) Edge properties with NaNs carry through when drop_na=False (note drop_na only for the keys anyway)
def test_edge_props_with_nans(builder_and_core, toy_nodes):
    bldr, core = builder_and_core
    # seed nodes
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    df = pd.DataFrame(
        {
            "source_id": ["A", "B", "C"],
            "source_layer": ["0", "0", "1"],
            "target_id": ["B", "C", "A"],
            "target_layer": ["0", "1", "0"],
            "w": [10.0, np.nan, 30.0],
        }
    )
    bldr.add_edges_from_dataframe(
        df,
        source_id_col="source_id",
        source_layer_col="source_layer",
        target_id_col="target_id",
        target_layer_col="target_layer",
        property_cols=["w"],
        drop_na=False,
    )
    ep = core.graph.ep["w"].a
    # ensure the nan surfaces
    assert np.isnan(ep[1])


# 22) Vertex category mapping must be stable when re-growing with new categories
def test_categorical_mapping_extends_not_restarts(builder_and_core):
    bldr, core = builder_and_core
    df1 = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"], "cat": ["x", "y"]})
    bldr.add_vertices_from_dataframe(df1, "node_id", "layer", property_cols=["cat"], drop_na=False)
    # remember original mapping
    orig = dict(core.vertex_categorical_mappings["cat"]["str_to_int"])

    # now add a new vertex with a new category "z"
    df2 = pd.DataFrame({"node_id": ["C"], "layer": ["0"], "cat": ["z"]})
    bldr.add_vertices_from_dataframe(df2, "node_id", "layer", property_cols=["cat"], drop_na=False)
    newmap = core.vertex_categorical_mappings["cat"]["str_to_int"]
    # existing x,y codes unchanged
    assert newmap["x"] == orig["x"]
    assert newmap["y"] == orig["y"]
    # new code for 'z' added
    assert "z" in newmap and newmap["z"] not in (orig["x"], orig["y"])


# 23) Vertex Property-name collision with internal vp keys should raise
def test_property_name_collision_with_internal(builder_and_core, toy_nodes):
    bldr, core = builder_and_core
    # 'layer_hash' is used internally
    df = toy_nodes.copy()
    df["layer_hash"] = [0, 1, 2]
    with pytest.raises(ValueError):
        bldr.add_vertices_from_dataframe(
            df, "node_id", "layer", property_cols=["layer_hash"], drop_na=False
        )


# 23) Edge Property-name collision with internal ep keys should raise
def test_edge_property_name_collision_with_internal(builder_and_core, toy_nodes):
    bldr, core = builder_and_core
    # seed the nodes
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)

    df = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "C"],
            "target_layer": ["0", "1"],
            "e_id": [0, 1],
        }
    )
    with pytest.raises(ValueError):
        bldr.add_edges_from_dataframe(
            df,
            source_id_col="source_id",
            source_layer_col="source_layer",
            target_id_col="target_id",
            target_layer_col="target_layer",
            property_cols=["e_id"],
            drop_na=False,
        )


# 22) Edge‐side category mapping must extend, not restart, when you re‐grow with new categories
def test_edge_categorical_mapping_extends_not_restarts(builder_and_core, toy_nodes):
    bldr, core = builder_and_core
    # first seed the nodes
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)

    # 1) ingest two edges with categories "x","y"
    df1 = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "C"],
            "target_layer": ["0", "1"],
            "cat": ["x", "y"],
        }
    )
    bldr.add_edges_from_dataframe(
        df1,
        source_id_col="source_id",
        source_layer_col="source_layer",
        target_id_col="target_id",
        target_layer_col="target_layer",
        property_cols=["cat"],
        drop_na=False,
    )
    orig = dict(core.edge_categorical_mappings["cat"]["str_to_int"])

    # 2) now add one more edge with a new category "z"
    df2 = pd.DataFrame(
        {
            "source_id": ["C"],
            "source_layer": ["1"],
            "target_id": ["A"],
            "target_layer": ["0"],
            "cat": ["z"],
        }
    )
    bldr.add_edges_from_dataframe(
        df2,
        source_id_col="source_id",
        source_layer_col="source_layer",
        target_id_col="target_id",
        target_layer_col="target_layer",
        property_cols=["cat"],
        drop_na=False,
    )
    newmap = core.edge_categorical_mappings["cat"]["str_to_int"]

    # existing codes must be unchanged...
    assert newmap["x"] == orig["x"]
    assert newmap["y"] == orig["y"]
    # ...and the new category got its own fresh code
    assert "z" in newmap and newmap["z"] not in (orig["x"], orig["y"])


# 24) summary() reports the correct counts on a mixed run
def test_summary_counts(builder_and_core, toy_nodes, toy_edges):
    bldr, core = builder_and_core
    # purposely drop one node & one edge
    n2 = pd.concat([toy_nodes, pd.DataFrame([{"node_id": None, "layer": "0"}])], ignore_index=True)
    e2 = pd.concat(
        [
            toy_edges,
            pd.DataFrame(
                [{"source_id": "A", "source_layer": "0", "target_id": None, "target_layer": "0"}]
            ),
        ],
        ignore_index=True,
    )
    bldr.grow_onion(
        n2,
        e2,
        node_prop_cols=[],
        edge_prop_cols=[],
        drop_na=True,
        drop_duplicates=True,
        verbose=False,
    )
    s = bldr.summary()
    # we dropped 1 node, dropped 1 edge invalid, final should be 3 nodes, 2 edges
    assert "dropped_na=1" in s
    assert "dropped_invalid=1" in s
    assert "final=3" in s.splitlines()[0]  # first line
    assert "final=2" in s.splitlines()[1]


# What happens when we introduce a brand-new layer mid-stream?
def test_layer_mapping_extends_with_new_layer(builder_and_core):
    bldr, core = builder_and_core
    # first ingest layer “0”
    df1 = pd.DataFrame({"node_id": ["A"], "layer": ["0"]})
    bldr.add_vertices_from_dataframe(df1, "node_id", "layer", drop_na=False)
    first_map = core._map_layer("0")
    # now ingest layer “2”
    df2 = pd.DataFrame({"node_id": ["B"], "layer": ["2"]})
    bldr.add_vertices_from_dataframe(df2, "node_id", "layer", drop_na=False)
    # ensure the new layer got a new integer code, not reusing “0”
    assert core._map_layer("2") != first_map


# We have have custom property_types override for nodes so lets do the same for edges
def test_edge_string_override_forces_categorical(builder_and_core, toy_nodes, toy_edges):
    bldr, core = builder_and_core
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    # strength is numeric, but we force it to be categorical
    bldr.add_edges_from_dataframe(
        toy_edges,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["strength"],
        drop_na=False,
        string_override=True,
    )
    assert "strength" in core.edge_categorical_mappings
    assert core.graph.ep["strength"].value_type().startswith("int")


# Testing edges where only one endpoint layer is new or missing, drop_na=False but invalid endpoints get filtered out
def test_edge_partial_invalid_layers(builder_and_core, toy_nodes):
    bldr, core = builder_and_core
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    df = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["0", "X"],  # “X” not seen before
            "target_id": ["B", "A"],
            "target_layer": ["0", "0"],
            "w": [1, 2],
        }
    )
    bldr.add_edges_from_dataframe(
        df,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["w"],
        drop_na=False,
    )
    # only the valid A→B edge should be added
    assert core.graph.num_edges() == 1


# If we feed in 10,000 distinct categories in one go, does our categorical mapping still scale (no memory leaks or integer overflows)?
@pytest.mark.slow
def test_huge_categorical_cardinality(builder_and_core, toy_nodes):
    bldr, core = builder_and_core
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    N = 10_000 * 10  # previously tested up to 10_000*1000
    cats = [f"C{i}" for i in range(N)]
    df = pd.DataFrame(
        {
            "source_id": ["A"] * N,
            "source_layer": ["0"] * N,
            "target_id": ["B"] * N,
            "target_layer": ["0"] * N,
            "cat": cats,
        }
    )
    bldr.add_edges_from_dataframe(
        df,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["cat"],
        drop_na=False,
        drop_duplicates=False,
    )
    # confirm we got N distinct codes
    assert len(core.edge_categorical_mappings["cat"]["str_to_int"]) == N


# Re-running the entire grow_onion on the same data with drop_duplicates=False should append vertices/edges but not corrupt existing mappings or reorder IDs.
# TODO: double check intended consequences of this
def test_grow_onion_idempotent(builder_and_core, toy_nodes, toy_edges):
    bldr, core = builder_and_core
    bldr.grow_onion(toy_nodes, toy_edges, drop_na=False, drop_duplicates=False)
    before_nodes = core.graph.num_vertices()
    before_edges = core.graph.num_edges()
    # run again
    bldr.grow_onion(toy_nodes, toy_edges, drop_na=False, drop_duplicates=False)
    # vertices and edges should have doubled
    assert core.graph.num_vertices() == before_nodes * 2
    assert core.graph.num_edges() == before_edges * 2
    # original mapping intact
    for (lay, nid), idx in core.custom_id_to_vertex_index.items():
        assert isinstance(idx, int)


def test_edge_property_alignment_after_filtering(builder_and_core):
    bldr, core = builder_and_core
    # seed nodes
    toy_nodes = pd.DataFrame({"node_id": ["A"], "layer": ["0"]})
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    # mixed df: second row has missing target, third has invalid layer
    df = pd.DataFrame(
        {
            "source_id": ["A", "A", "A"],
            "source_layer": ["0", "0", "X"],
            "target_id": ["A", None, "A"],
            "target_layer": ["0", "0", "0"],
            "w": [1, 2, 3],
            "cat": ["x", "y", "z"],
        }
    )
    bldr.add_edges_from_dataframe(
        df,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["w", "cat"],
        drop_na=True,
    )
    # only the first row remains
    assert core.graph.num_edges() == 1
    # both properties exist and length==1
    assert len(core.graph.ep["w"].a) == 1
    assert len(core.graph.ep["cat"].a) == 1
    # and the single cat‐code maps back to "x"
    code = core.graph.ep["cat"].a[0]
    assert core.edge_categorical_mappings["cat"]["int_to_str"][code] == "x"


def test_two_builders_same_core():
    core = OnionNetGraph()
    b1 = OnionNetBuilder(core)
    b2 = OnionNetBuilder(core)
    df_n = pd.DataFrame({"node_id": ["A"], "layer": ["0"]})
    df_e = pd.DataFrame(
        {
            "source_id": ["A"],
            "source_layer": ["0"],
            "target_id": ["A"],
            "target_layer": ["0"],
            "w": [5],
        }
    )
    b1.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    b2.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["w"],
        drop_na=False,
    )
    assert core.graph.num_vertices() == 1
    assert core.graph.num_edges() == 1


def test_property_name_is_string_number(builder_and_core):
    bldr, core = builder_and_core
    df = pd.DataFrame({"node_id": ["A"], "layer": ["0"], "1": [100]})
    # either error or treat it as "prop_1" internally—but must not crash
    bldr.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=["1"], drop_na=False)
    assert "1" in core.graph.vp
    assert core.graph.vp["1"].a[0] == 100


def test_layer_key_normalization(builder_and_core):
    bldr, core = builder_and_core
    df = pd.DataFrame({"node_id": ["X", "Y", "Z"], "layer": ["1", " 1", "1 "]})
    bldr.add_vertices_from_dataframe(df, "node_id", "layer", drop_na=False)
    # Layers differing only by whitespace are treated as distinct
    codes = {core._map_layer(l) for l in ["1", " 1", "1 "]}
    assert len(codes) == 3


def test_edge_property_type_conflict(builder_and_core, toy_nodes, toy_edges):
    bldr, core = builder_and_core
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    # 1) ingest strength as numeric
    bldr.add_edges_from_dataframe(
        toy_edges,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["strength"],
        drop_na=False,
        property_types={"strength": "int"},
    )
    # 2) try to ingest strength again as categorical → should error
    with pytest.raises(ValueError):
        bldr.add_edges_from_dataframe(
            toy_edges,
            "source_id",
            "source_layer",
            "target_id",
            "target_layer",
            property_cols=["strength"],
            drop_na=False,
            property_types={"strength": "str"},
        )


def test_grow_onion_idempotent_with_drop_duplicates(builder_and_core, toy_nodes, toy_edges):
    bldr, core = builder_and_core
    bldr.grow_onion(toy_nodes, toy_edges, drop_na=False, drop_duplicates=True)
    v1, e1 = core.graph.num_vertices(), core.graph.num_edges()
    bldr.grow_onion(toy_nodes, toy_edges, drop_na=False, drop_duplicates=True)
    assert core.graph.num_vertices() == v1
    assert core.graph.num_edges() == e1
