import pytest
import numpy as np
from onionnet.core import OnionNetGraph
from onionnet.builder import OnionNetBuilder
from onionnet.searcher import OnionNetSearcher
from graph_tool.all import GraphView, shortest_distance
import pandas as pd
import graph_tool.all as gt


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def simple_graph():
    """Tiny 2-node, 1-edge graph with a numeric vertex prop 'score'."""
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"], "score": [10, 20]})
    b.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=["score"], drop_na=False)
    return core


@pytest.fixture
def builder_and_core():
    core = OnionNetGraph()
    builder = OnionNetBuilder(core)
    return builder, core


@pytest.fixture
def chain_graph_and_searcher():
    # Build a simple directed chain A→B→C
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    df_nodes = pytest.importorskip("pandas").DataFrame(
        {"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"]}
    )
    b.add_vertices_from_dataframe(df_nodes, "node_id", "layer", drop_na=False)
    df_edges = pytest.importorskip("pandas").DataFrame(
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


#### Basic Tests ####


def test_compute_on_shortest_basic(chain_graph_and_searcher):
    """
    A→B→C, shortest from A to C should include A,B,C.
    """
    core, s = chain_graph_and_searcher
    # indices 0,1,2 correspond to A,B,C
    on_sp = s.compute_on_shortest(0, [2], inplace=False, return_gv=False)
    # check property map: True for all three
    vals = [bool(on_sp[v]) for v in core.graph.vertices()]
    assert vals == [True, True, True]


def test_compute_on_shortest_graphview(chain_graph_and_searcher):
    """
    When return_gv=True, we get a GraphView containing only the on-shortest-path vertices.
    """
    core, s = chain_graph_and_searcher
    gv = s.compute_on_shortest(0, [2], return_gv=True)
    assert isinstance(gv, GraphView)
    # should have 3 vertices still (all on path) and 2 edges
    assert gv.num_vertices() == 3
    assert gv.num_edges() == 2


def test_compute_on_shortest_invalid_indices(chain_graph_and_searcher):
    """
    Passing an out-of-range source or target index should raise ValueError.
    """
    _, s = chain_graph_and_searcher
    with pytest.raises(ValueError):
        s.compute_on_shortest(99, [2])
    with pytest.raises(ValueError):
        s.compute_on_shortest(0, [99])


def test_search_downstream(chain_graph_and_searcher):
    """
    Downstream search from A with max_dist=1 should include A and B only.
    """
    core, s = chain_graph_and_searcher
    gv = s.search(start_node_idx=0, max_dist=1, direction="downstream", show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0, 1}


def test_search_upstream(chain_graph_and_searcher):
    """
    Upstream search from C with max_dist=1 should include C and B only.
    """
    core, s = chain_graph_and_searcher
    gv = s.search(start_node_idx=2, max_dist=1, direction="upstream", show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {1, 2}


def test_search_bidirectional(chain_graph_and_searcher):
    """
    Bidirectional search from B max_dist=1 should include A,B,C.
    """
    core, s = chain_graph_and_searcher
    gv = s.search(start_node_idx=1, max_dist=1, direction="bi", show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0, 1, 2}


def test_search_any(chain_graph_and_searcher):
    """
    'any' search treats the graph as undirected.
    From C with max_dist=2 in an undirected chain, should reach A,B,C.
    """
    core, s = chain_graph_and_searcher
    gv = s.search(start_node_idx=2, max_dist=2, direction="any", show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0, 1, 2}


def test_bi_vs_any_distinction(builder_and_core):
    """
    In this “V-shape” graph (A→B and C→B), a bidirectional ('bi') search
    from A with max_dist=2 will only see A→B (so {A,B}), because there is no
    directed path A→B→C.  But an undirected ('any') search at max_dist=2
    will go A–B–C and thus pick up C as well.
    """
    bldr, core = builder_and_core

    # 1) Add our three nodes A, B, C all on layer "0"
    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)

    # 2) Add two directed edges: A→B and C→B
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "C"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "B"],
            "target_layer": ["0", "0"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        source_id_col="source_id",
        source_layer_col="source_layer",
        target_id_col="target_id",
        target_layer_col="target_layer",
        property_cols=None,
        drop_na=False,
    )

    s = OnionNetSearcher(core)

    # BIDIRECTIONAL: union of upstream(from A) and downstream(from A)
    # downstream(A) = {A,B}, upstream(A) = {A}  →  {A,B}
    gv_bi = s.search(start_node_idx=0, max_dist=2, direction="bi", show_plot=False)
    got_bi = {int(v) for v in gv_bi.vertices()}
    assert got_bi == {0, 1}, f"expected just A,B for 'bi', got {got_bi}"

    # ANY (undirected): can traverse A–B–C in two hops → {A,B,C}
    gv_any = s.search(start_node_idx=0, max_dist=2, direction="any", show_plot=False)
    got_any = {int(v) for v in gv_any.vertices()}
    assert got_any == {0, 1, 2}, f"expected A,B,C for 'any', got {got_any}"


def test_view_layers_and_filter(chain_graph_and_searcher):
    """
    view_layers should filter vertices by layer names.
    Here all nodes are layer '0', so view_layers('0') returns the full graph.
    """
    core, s = chain_graph_and_searcher
    gv = s.view_layers("0")
    assert gv.num_vertices() == 3
    # passing return_filter=True returns the bool PropertyMap
    vfilt = s.view_layers(["0"], return_filter=True)
    assert hasattr(vfilt, "get_array")


def test_view_layers_invalid(chain_graph_and_searcher):
    """
    Asking for a non-existent layer should ValueError.
    """
    _, s = chain_graph_and_searcher
    with pytest.raises(ValueError):
        s.view_layers("nonexistent")


@pytest.mark.parametrize(
    "connectivity, expected_size",
    [
        ("strong", 0),  # no strongly‐connected components of size ≥ 2 in a simple chain
        ("weak", 3),  # undirected (weak) connectivity groups the whole chain into one component
    ],
)
def test_view_components(chain_graph_and_searcher, connectivity, expected_size):
    """
    view_components should respect both strong and weak connectivity modes.
    With size_threshold=2 on our A→B→C chain:
      - strong: each vertex is its own SCC ⇒ 0 comps of size ≥2 ⇒  0 vertices
      - weak:  the entire chain is one comp  ⇒ 1 comp of size 3 ≥2 ⇒  3 vertices
    And with threshold=4 there are no components large enough in either mode.
    """
    core, searcher = chain_graph_and_searcher

    # first with threshold=2
    gv = searcher.view_components(size_threshold=2, connectivity=connectivity)
    assert gv.num_vertices() == expected_size

    # now with threshold=4 (too high for any mode)
    gv2 = searcher.view_components(size_threshold=4, connectivity=connectivity)
    assert gv2.num_vertices() == 0


def test_filter_view_by_property_vertex(simple_graph):
    """
    filter_view_by_property for the vertex dimension should
    correctly keep only nodes whose 'score' > 15.
    """
    core = simple_graph
    searcher = OnionNetSearcher(core)

    gv = searcher.filter_view_by_property(
        prop_name="score", target_value=15, comparison=">", dim="v"
    )

    remaining = {int(v) for v in gv.vertices()}
    assert remaining == {1}  # only the second node (score=20)


def test_filter_view_by_property_edge(chain_graph_and_searcher):
    """
    filter_view_by_property for edges should work and, if prune_isolated=True,
    drop any vertices that would otherwise become isolated.
    """
    core, searcher = chain_graph_and_searcher
    # attach a numeric 'w' property to the two chain edges
    wmap = core.graph.new_edge_property("int")
    for e, val in zip(core.graph.edges(), [5, 10]):
        wmap[e] = val
    core.graph.ep["w"] = wmap

    gv = searcher.filter_view_by_property(
        prop_name="w", target_value=[10], dim="e", prune_isolated=True
    )

    # Only one edge (the one with weight=10) remains,
    # and its two incident nodes must both survive.
    assert gv.num_edges() == 1
    assert gv.num_vertices() == 2


def test_compose_filters_and_return_prop(chain_graph_and_searcher):
    """
    compose_filters can combine predicates and return a PropertyMap.
    """
    core, s = chain_graph_and_searcher
    # predicate: nodes A or C only
    funcs = [lambda v: int(v) == 0, lambda v: int(v) == 2]
    prop = s.compose_filters(funcs, mode="or", type="v", return_prop=True)
    arr = [bool(prop[v]) for v in core.graph.vertices()]
    assert arr == [True, False, True]
    # as GraphView
    gv = s.compose_filters(funcs, mode="and", type="v", return_prop=False)
    assert isinstance(gv, GraphView)


def test_create_bipartite_gv(builder_and_core):
    """
    create_bipartite_gv should only keep edges between two specified layers.
    """
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    df = pytest.importorskip("pandas").DataFrame(
        {"node_id": ["A", "B", "C", "D"], "layer": ["L1", "L2", "L1", "L2"]}
    )
    b.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=None, drop_na=False)
    # encode 'layer' as string vp
    lbl = core.graph.new_vertex_property("string")
    for v in core.graph.vertices():
        lbl[v] = core.layer_code_to_name[core.graph.vp["layer_hash"][v]]
    core.graph.vp["layer_decoded"] = lbl

    # edges crossing and within layers
    df_e = pytest.importorskip("pandas").DataFrame(
        {
            "source_id": ["A", "A", "C"],
            "source_layer": ["L1", "L1", "L1"],
            "target_id": ["B", "C", "B"],
            "target_layer": ["L2", "L1", "L2"],
        }
    )
    b.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    s = OnionNetSearcher(core)
    gv = s.create_bipartite_gv("L1", "L2", "layer_decoded")
    # only edges A→B and C→B remain (cross-layer), vertices A,B,C remain
    assert gv.num_edges() == 2
    assert gv.num_vertices() == 3


#### Intermediate Tests ####


def test_search_invalid_direction(chain_graph_and_searcher):
    """
    Attempting a search with an unsupported direction string
    should raise ValueError.
    """
    _, s = chain_graph_and_searcher
    with pytest.raises(ValueError):
        s.search(start_node_idx=0, max_dist=1, direction="sideways", show_plot=False)


def test_filter_view_by_property_invalid_prop(simple_graph):
    """
    filter_view_by_property should raise ValueError when asked
    for a non-existent property (both vertex and edge) or invalid comparison.
    """
    core = simple_graph
    s = OnionNetSearcher(core)
    # vertex side invalid prop
    with pytest.raises(ValueError):
        s.filter_view_by_property("does_not_exist", 1, dim="v")
    # edge side invalid prop
    with pytest.raises(ValueError):
        s.filter_view_by_property("does_not_exist", 1, dim="e")
    # invalid comparison operator
    with pytest.raises(ValueError):
        s.filter_view_by_property("score", 1, comparison="<>", dim="v")


def test_view_layers_copy_returns_graph_not_view(chain_graph_and_searcher):
    """
    view_layers(copy_gv=True) should return a fresh Graph object,
    and modifying it must not mutate the original core.graph.
    """
    core, s = chain_graph_and_searcher
    gv = s.view_layers("0", copy_gv=True)
    from graph_tool.all import Graph

    assert isinstance(gv, Graph)
    original_nv = core.graph.num_vertices()
    gv.add_vertex()
    assert core.graph.num_vertices() == original_nv


def test_compute_on_shortest_inplace_removes_artificial(chain_graph_and_searcher):
    """
    When compute_on_shortest(inplace=True) is used, the artificial vertex
    added internally must be cleaned up—core.graph.num_vertices() remains unchanged.
    """
    core, s = chain_graph_and_searcher
    n_before = core.graph.num_vertices()
    _ = s.compute_on_shortest(0, [2], inplace=True, return_gv=False)
    assert core.graph.num_vertices() == n_before


def test_search_with_custom_graph_parameter(chain_graph_and_searcher):
    """
    Passing a custom GraphView as `g` into search should limit
    the search space to that view (e.g. prefilter out C on A→B→C).
    """
    core, s = chain_graph_and_searcher
    view_ab = GraphView(core.graph, vfilt=lambda v: int(v) in {0, 1})
    gv = s.search(start_node_idx=0, max_dist=2, direction="any", g=view_ab, show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0, 1}


@pytest.mark.parametrize("mode", ["xor", "not"])
def test_compose_filters_bad_mode(chain_graph_and_searcher, mode):
    """
    compose_filters should reject unsupported combination modes
    and raise ValueError.
    """
    _, s = chain_graph_and_searcher
    with pytest.raises(ValueError):
        s.compose_filters([lambda v: True], mode=mode, type="v")


def test_view_components_threshold_one(chain_graph_and_searcher):
    """
    view_components(size_threshold=1) under strong connectivity
    on a 3‐node chain should keep all singletons → 3 vertices.
    """
    core, s = chain_graph_and_searcher
    gv = s.view_components(size_threshold=1, connectivity="strong")
    assert gv.num_vertices() == 3


def test_filter_view_by_property_edge_no_prune(chain_graph_and_searcher):
    """
    filter_view_by_property on edges with prune_isolated=False
    should remove unmatched edges but keep all original vertices.
    """
    core, s = chain_graph_and_searcher
    wmap = core.graph.new_edge_property("int")
    for e in core.graph.edges():
        wmap[e] = 1
    core.graph.ep["w"] = wmap
    gv = s.filter_view_by_property("w", 999, dim="e", prune_isolated=False)
    assert gv.num_edges() == 0
    assert gv.num_vertices() == core.graph.num_vertices()


def test_create_bipartite_gv_no_cross_edges(
    builder_and_core,
):  # TODO: check if semantically correct
    """
    create_bipartite_gv on a graph with two layers but no edges
    should return an empty GraphView (0 vertices, 0 edges).
    """
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A", "B"], "layer": ["X", "Y"]})
    b.add_vertices_from_dataframe(df, "node_id", "layer", drop_na=False)
    s = OnionNetSearcher(core)
    gv = s.create_bipartite_gv("X", "Y", "layer_decoded")  # in practice would use layer_decoded
    assert gv.num_vertices() == 0 and gv.num_edges() == 0


def test_create_bipartite_gv_invalid_prop(builder_and_core):  # TODO: check if semantically correct
    """
    create_bipartite_gv should raise KeyError when asked
    to use a non-existent vertex property for filtering.
    """
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A", "B"], "layer": ["X", "Y"]})
    b.add_vertices_from_dataframe(df, "node_id", "layer", drop_na=False)
    s = OnionNetSearcher(core)
    with pytest.raises(KeyError):
        s.create_bipartite_gv("X", "Y", "does_not_exist")


# --- Tests for filter_edges_between_categories ------------------------------


def test_filter_edges_between_categories_basic(builder_and_core):
    """
    Only edges whose source_layer == L1 and target_layer == L2 should survive,
    and only their endpoints should remain.
    """
    from onionnet.searcher import OnionNetSearcher

    bldr, core = builder_and_core

    # 1) Add three nodes: A@L1, B@L2, C@L1
    import pandas as pd

    df_n = pd.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "layer": ["L1", "L2", "L1"],
        }
    )
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)

    # 2) Add three edges:
    #    A(L1)→B(L2), C(L1)→B(L2)  (should match)
    #    A(L1)→C(L1)               (should not match)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "C", "A"],
            "source_layer": ["L1", "L1", "L1"],
            "target_id": ["B", "B", "C"],
            "target_layer": ["L2", "L2", "L1"],
        }
    )
    # we need source_layer and target_layer as edge props so filter can see them
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["source_layer", "target_layer"],
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    gv = s.filter_edges_between_categories(source_label="L1", target_label="L2")

    from graph_tool.all import GraphView

    assert isinstance(gv, GraphView)

    # Expect only the first two edges (A→B and C→B)
    result_edges = {(int(e.source()), int(e.target())) for e in gv.edges()}
    # node ordering is exactly in the order they were added: A=0, B=1, C=2
    assert result_edges == {(0, 1), (2, 1)}

    # Endpoints A, B, C should survive; no other vertices
    assert set(int(v) for v in gv.vertices()) == {0, 1, 2}


def test_filter_edges_between_categories_unknown_label_raises(builder_and_core):
    """
    Asking to filter by a label that doesn't exist in the mapping should KeyError.
    """
    from onionnet.searcher import OnionNetSearcher
    import pandas as pd

    bldr, core = builder_and_core
    # add minimal nodes & one edge
    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["L1", "L2"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A"],
            "source_layer": ["L1"],
            "target_id": ["B"],
            "target_layer": ["L2"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["source_layer", "target_layer"],
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    # unknown source_label
    with pytest.raises(KeyError):
        s.filter_edges_between_categories(source_label="DOES_NOT_EXIST", target_label="L2")
    # unknown target_label
    with pytest.raises(KeyError):
        s.filter_edges_between_categories(source_label="L1", target_label="XXX")


# --- Tests for the new filter_edges and _prune_isolated helpers -----


def test_filter_edges_return_prop_only(builder_and_core):
    """
    If return_view=False, filter_edges should hand back the raw EdgePropertyMap
    and not a GraphView.
    """
    from onionnet.searcher import OnionNetSearcher

    bldr, core = builder_and_core
    # A→B→C chain
    import pandas as pd

    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "C"],
            "target_layer": ["0", "0"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    # predicate: only keep the B→C edge (index 1)
    prop = s.filter_edges(lambda e: int(e.source()) == 1, return_view=False)
    assert not isinstance(prop, type(core.graph))  # raw property, not GraphView
    # exactly one True entry in property map
    kept = [prop[e] for e in core.graph.edges()]
    assert sum(kept) == 1 and kept == [False, True]


def test_filter_edges_prunes_isolated(builder_and_core):
    """
    filter_edges should drop any isolated vertices after edge filtering.
    """
    from onionnet.searcher import OnionNetSearcher

    bldr, core = builder_and_core
    # star graph: 0→1, 0→2
    import pandas as pd

    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "A"],
            "source_layer": ["0", "0"],
            "target_id": ["B", "C"],
            "target_layer": ["0", "0"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    s = OnionNetSearcher(core)

    # keep only the A→B edge
    gv = s.filter_edges(lambda e: int(e.target()) == 1, return_view=True)
    # B and A survive, but C (vertex 2) should have been pruned
    assert set(int(v) for v in gv.vertices()) == {0, 1}
    assert set((int(e.source()), int(e.target())) for e in gv.edges()) == {(0, 1)}


def test_filter_edges_no_matches_yields_empty(builder_and_core):
    """
    If predicate never returns True, filter_edges should give an empty GraphView
    (no edges, no vertices).
    """
    from onionnet.searcher import OnionNetSearcher

    bldr, core = builder_and_core
    # tiny 2‐node graph
    import pandas as pd

    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["0", "0"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A"],
            "source_layer": ["0"],
            "target_id": ["B"],
            "target_layer": ["0"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    s = OnionNetSearcher(core)

    gv = s.filter_edges(lambda e: False, return_view=True)
    assert gv.num_edges() == 0
    assert gv.num_vertices() == 0


# --- Tests for filter_edges_between_categories edge‐cases -----


def test_filter_between_categories_mismatched_props(builder_and_core):
    """
    Even if no edge-level props exist, filtering by two valid layer labels
    should still keep the A@X→B@Y edge.
    """
    from onionnet.searcher import OnionNetSearcher
    import pandas as pd

    bldr, core = builder_and_core
    # minimal two‐node one‐edge
    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["X", "Y"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)

    # add the edge (but with no extra props)
    df_e = pd.DataFrame(
        {
            "source_id": ["A"],
            "source_layer": ["X"],
            "target_id": ["B"],
            "target_layer": ["Y"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    # X→Y is a valid layer‐pair, so the single edge should survive
    gv = s.filter_edges_between_categories("X", "Y")
    assert gv.num_edges() == 1
    assert {int(v) for v in gv.vertices()} == {0, 1}


def test_filter_between_categories_reversed(builder_and_core):
    """
    If called with swapped source/target labels, the predicate should only pick
    the edges in that exact direction.
    """
    from onionnet.searcher import OnionNetSearcher

    bldr, core = builder_and_core
    # two‐node graph, A@L1→B@L2
    import pandas as pd

    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["L1", "L2"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A"],
            "source_layer": ["L1"],
            "target_id": ["B"],
            "target_layer": ["L2"],
        }
    )
    # add layer codes as props
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["source_layer", "target_layer"],
        drop_na=False,
    )
    s = OnionNetSearcher(core)

    gv_forward = s.filter_edges_between_categories("L1", "L2")
    gv_backward = s.filter_edges_between_categories("L2", "L1")
    assert gv_forward.num_edges() == 1
    assert gv_backward.num_edges() == 0


# --- Tests for create_bipartite_gv isolated-vertex pruning -----


def test_create_bipartite_prune_isolated(builder_and_core):
    """
    create_bipartite_gv should also prune vertices that end up isolated
    (e.g. if only one endpoint matches).
    """
    from onionnet.searcher import OnionNetSearcher

    bldr, core = builder_and_core
    # three nodes of two layers, but only one cross‐edge
    import pandas as pd

    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["L1", "L2", "L1"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    # assign layer_decoded
    lbl = core.graph.new_vertex_property("string")
    for v in core.graph.vertices():
        lbl[v] = core.layer_code_to_name[core.graph.vp["layer_hash"][v]]
    core.graph.vp["layer_decoded"] = lbl

    df_e = pd.DataFrame(
        {
            "source_id": ["A", "C"],
            "source_layer": ["L1", "L1"],
            "target_id": ["B", "A"],
            "target_layer": ["L2", "L1"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    s = OnionNetSearcher(core)

    gv = s.create_bipartite_gv("L1", "L2", "layer_decoded")
    # only A→B should survive, C and the self‐edge on A→A pruned
    assert gv.num_edges() == 1
    assert set(int(v) for v in gv.vertices()) == {0, 1}


def test_filter_edges_between_categories_modes(builder_and_core):
    """
    Building a 3-node graph A@L1→B@L2 and B@L2→C@L1:
      - forward('L1','L2')  should yield only A→B
      - reverse('L1','L2')  should yield only B→C
      - both('L1','L2')     should yield both edges
    """
    bldr, core = builder_and_core
    # 1) nodes A@L1, B@L2, C@L1
    df_n = pd.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "layer": ["L1", "L2", "L1"],
        }
    )
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)

    # 2) two edges: A(L1)->B(L2) and B(L2)->C(L1)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["L1", "L2"],
            "target_id": ["B", "C"],
            "target_layer": ["L2", "L1"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["source_layer", "target_layer"],
        drop_na=False,
    )

    s = OnionNetSearcher(core)

    # forward: only A->B
    gv_fwd = s.filter_edges_between_categories("L1", "L2", mode="forward")
    edges_fwd = {(int(e.source()), int(e.target())) for e in gv_fwd.edges()}
    assert edges_fwd == {(0, 1)}
    verts_fwd = {int(v) for v in gv_fwd.vertices()}
    assert verts_fwd == {0, 1}

    # reverse: only B->C
    gv_rev = s.filter_edges_between_categories("L1", "L2", mode="reverse")
    edges_rev = {(int(e.source()), int(e.target())) for e in gv_rev.edges()}
    assert edges_rev == {(1, 2)}
    verts_rev = {int(v) for v in gv_rev.vertices()}
    assert verts_rev == {1, 2}

    # both: both edges
    gv_both = s.filter_edges_between_categories("L1", "L2", mode="both")
    edges_both = {(int(e.source()), int(e.target())) for e in gv_both.edges()}
    assert edges_both == {(0, 1), (1, 2)}
    verts_both = {int(v) for v in gv_both.vertices()}
    assert verts_both == {0, 1, 2}


def test_create_bipartite_gv_is_symmetric(builder_and_core):
    """
    create_bipartite_gv('L1','L2') and create_bipartite_gv('L2','L1')
    should yield identical sets of nodes & edges (since it uses mode='both').
    """
    bldr, core = builder_and_core
    # nodes A@L1, B@L2, C@L1
    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["L1", "L2", "L1"]})
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    # edges A(L1)->B(L2) and B(L2)->C(L1)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["L1", "L2"],
            "target_id": ["B", "C"],
            "target_layer": ["L2", "L1"],
        }
    )
    bldr.add_edges_from_dataframe(
        df_e,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    gv12 = s.create_bipartite_gv("L1", "L2")
    gv21 = s.create_bipartite_gv("L2", "L1")

    # compare edge‐sets
    e12 = {(int(e.source()), int(e.target())) for e in gv12.edges()}
    e21 = {(int(e.source()), int(e.target())) for e in gv21.edges()}
    assert e12 == e21 == {(0, 1), (1, 2)}

    # compare vertex‐sets
    v12 = {int(v) for v in gv12.vertices()}
    v21 = {int(v) for v in gv21.vertices()}
    assert v12 == v21 == {0, 1, 2}


#### Tests for compute_on_shortest with directed/undirected shortcuts ----


def test_on_shortest_directed_vs_undirected_shortcut():
    # A->B->C->D, plus a reverse edge D->B
    # Directed A→D shortest: A-B-C-D (len=3) → {A,B,C,D}
    # Undirected shortest: A-B-D via B–D (len=2) → {A,B,D}
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    b.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A", "B", "C", "D"], "layer": ["0"] * 4}),
        "node_id",
        "layer",
        drop_na=False,
    )
    b.add_edges_from_dataframe(
        pd.DataFrame(
            {
                "source_id": ["A", "B", "C", "D"],
                "source_layer": ["0", "0", "0", "0"],
                "target_id": ["B", "C", "D", "B"],  # reverse-only shortcut
                "target_layer": ["0", "0", "0", "0"],
            }
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    gv_dir = s.compute_on_shortest(source=0, targets=[3], directed=True, return_gv=True)
    gv_undir = s.compute_on_shortest(source=0, targets=[3], directed=False, return_gv=True)

    assert {int(v) for v in gv_dir.vertices()} == {0, 1, 2, 3}
    assert {int(v) for v in gv_undir.vertices()} == {0, 1, 3}


def test_on_shortest_reverse_only_path_unreachable_directed():
    # Only edges C->B and B->A; from A to C:
    # directed=True: unreachable → empty view
    # directed=False: A–B–C reachable → {A,B,C}
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    b.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["0", "0", "0"]}),
        "node_id",
        "layer",
        drop_na=False,
    )
    b.add_edges_from_dataframe(
        pd.DataFrame(
            {
                "source_id": ["C", "B"],
                "source_layer": ["0", "0"],
                "target_id": ["B", "A"],
                "target_layer": ["0", "0"],
            }
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    gv_dir = s.compute_on_shortest(source=0, targets=[2], directed=True, return_gv=True)
    gv_undir = s.compute_on_shortest(source=0, targets=[2], directed=False, return_gv=True)

    assert sum(1 for _ in gv_dir.vertices()) == 0
    assert {int(v) for v in gv_undir.vertices()} == {0, 1, 2}


def test_on_shortest_multi_target_undirected_union():
    # Chain A-B-C-D plus E attached to C.
    # From A to {D,E} undirected: union {A,B,C,D,E}
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    b.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A", "B", "C", "D", "E"], "layer": ["0"] * 5}),
        "node_id",
        "layer",
        drop_na=False,
    )
    b.add_edges_from_dataframe(
        pd.DataFrame(
            {
                "source_id": ["A", "B", "C", "C"],
                "source_layer": ["0", "0", "0", "0"],
                "target_id": ["B", "C", "D", "E"],
                "target_layer": ["0", "0", "0", "0"],
            }
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )

    s = OnionNetSearcher(core)
    gv = s.compute_on_shortest(source=0, targets=[3, 4], directed=False, return_gv=True)
    assert {int(v) for v in gv.vertices()} == {0, 1, 2, 3, 4}


def test_on_shortest_accepts_name_tuples_and_directed_flag():
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    b.add_vertices_from_dataframe(
        pd.DataFrame({"node_id": ["A", "B"], "layer": ["L", "L"]}),
        "node_id",
        "layer",
        drop_na=False,
    )
    b.add_edges_from_dataframe(
        pd.DataFrame(
            {
                "source_id": ["A"],
                "source_layer": ["L"],
                "target_id": ["B"],
                "target_layer": ["L"],
            }
        ),
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=None,
        drop_na=False,
    )
    s = OnionNetSearcher(core)
    gv = s.compute_on_shortest(
        source=("L", "A"), targets=[("L", "B")], directed=False, return_gv=True
    )
    assert {int(v) for v in gv.vertices()} == {0, 1}


# ---- Additional comprehensive tests ----


def make_simple_core():
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    df_nodes = pd.DataFrame(
        {
            "node_id": ["A", "B", "C", "D"],
            "layer": ["L", "L", "R", "R"],
            "val": [0, 1, 2, 3],
        }
    )
    b.add_vertices_from_dataframe(
        df_nodes, "node_id", "layer", property_cols=["val"], drop_na=False
    )

    df_edges = pd.DataFrame(
        {
            "source_id": ["A", "B", "C", "D"],
            "source_layer": ["L", "L", "R", "R"],
            "target_id": ["B", "C", "D", "A"],
            "target_layer": ["L", "R", "R", "L"],
            "w": [1, 2, 3, 4],
        }
    )
    b.add_edges_from_dataframe(
        df_edges,
        "source_id",
        "source_layer",
        "target_id",
        "target_layer",
        property_cols=["w"],
        drop_na=False,
    )
    return core


def test_compute_on_shortest_unreachable_and_undirected():
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    df_nodes = pd.DataFrame({"node_id": ["X", "Y"], "layer": ["L", "L"]})
    b.add_vertices_from_dataframe(df_nodes, "node_id", "layer", drop_na=False)
    s = OnionNetSearcher(core)

    # Unreachable target (no edge between X and Y) → empty view
    gv_empty = s.compute_on_shortest(0, [1], return_gv=True)
    assert gv_empty.num_vertices() == 0

    # Undirected on a chain A-B-C
    core2 = OnionNetGraph(directed=True)
    b2 = OnionNetBuilder(core2)
    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["L", "L", "L"]})
    b2.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["L", "L"],
            "target_id": ["B", "C"],
            "target_layer": ["L", "L"],
        }
    )
    b2.add_edges_from_dataframe(
        df_e, "source_id", "source_layer", "target_id", "target_layer", drop_na=False
    )
    s2 = OnionNetSearcher(core2)
    gv_dir = s2.compute_on_shortest(0, [2], return_gv=True, directed=True)
    assert gv_dir.num_vertices() == 3
    gv_undir = s2.compute_on_shortest(2, [0], return_gv=True, directed=False)
    assert gv_undir.num_vertices() == 3


def test__bfs_traversal_modes_and_error():
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["L", "L", "L"]})
    b.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "B"],
            "source_layer": ["L", "L"],
            "target_id": ["B", "C"],
            "target_layer": ["L", "L"],
        }
    )
    b.add_edges_from_dataframe(
        df_e, "source_id", "source_layer", "target_id", "target_layer", drop_na=False
    )
    s = OnionNetSearcher(core)
    g = core.graph
    vf = g.new_vertex_property("bool")
    ef = g.new_edge_property("bool")
    s._bfs_traversal([g.vertex(0)], vf, ef, mode="downstream")
    assert vf[g.vertex(0)] and vf[g.vertex(1)] and vf[g.vertex(2)]
    vf2 = g.new_vertex_property("bool")
    ef2 = g.new_edge_property("bool")
    s._bfs_traversal([g.vertex(2)], vf2, ef2, mode="upstream")
    assert vf2[g.vertex(0)] and vf2[g.vertex(1)] and vf2[g.vertex(2)]
    with pytest.raises(ValueError):
        s._bfs_traversal([g.vertex(0)], vf, ef, mode="sideways")


def test_filter_view_by_property_vertices_and_edges():
    core = make_simple_core()
    s = OnionNetSearcher(core)
    g = core.graph
    gv_v = s.filter_view_by_property("val", 2, comparison=">=", dim="v", prune_isolated=False)
    vs = {int(v) for v in gv_v.vertices()}
    assert vs == {2, 3}
    gv_v2 = s.filter_view_by_property("val", {0, 1, 2, 3}, dim="v", prune_isolated=True)
    assert isinstance(gv_v2, gt.GraphView)
    gv_e = s.filter_view_by_property("w", {2, 4}, dim="e", prune_isolated=True)
    assert all((v.out_degree() + v.in_degree()) > 0 for v in gv_e.vertices())
    with pytest.raises(ValueError):
        s.filter_view_by_property("nope", 1, dim="v")
    with pytest.raises(ValueError):
        s.filter_view_by_property("nope", 1, dim="e")


def test_compose_filters_and_modes():
    core = make_simple_core()
    s = OnionNetSearcher(core)
    g = core.graph
    f_gt0 = lambda v: g.vp["val"][v] > 0
    f_even = lambda v: (g.vp["val"][v] % 2) == 0
    gv = s.compose_filters([f_gt0, f_even], mode="and", type="v")
    vals = [g.vp["val"][v] for v in gv.vertices()]
    assert set(map(int, vals)) == {2}
    pm = s.compose_filters([f_gt0, f_even], mode="or", type="v", return_prop=True)
    assert hasattr(pm, "a")
    assert pm.a.sum() >= 1
    e_ge3 = lambda e: g.ep["w"][e] >= 3
    gv2 = s.compose_filters([e_ge3], type="e")
    assert all(g.ep["w"][e] >= 3 for e in gv2.edges())
    with pytest.raises(ValueError):
        s.compose_filters([f_gt0], mode="xor")
    with pytest.raises(ValueError):
        s.compose_filters([f_gt0], type="x")


def test_filter_edges_between_categories_and_bipartite():
    core = make_simple_core()
    s = OnionNetSearcher(core)
    gv_fwd = s.filter_edges_between_categories("L", "R", mode="forward")
    assert gv_fwd.num_edges() >= 1
    gv_rev = s.filter_edges_between_categories("L", "R", mode="reverse")
    assert gv_rev.num_edges() >= 1
    gv_both = s.filter_edges_between_categories("L", "R", mode="both")
    assert gv_both.num_edges() == gv_fwd.num_edges() + gv_rev.num_edges()
    with pytest.raises(KeyError):
        s.filter_edges_between_categories("X", "R")
    with pytest.raises(ValueError):
        s.filter_edges_between_categories("L", "R", mode="nope")
    gv_bi = s.create_bipartite_gv("L", "R")
    assert isinstance(gv_bi, gt.GraphView)
    with pytest.raises(KeyError):
        s.create_bipartite_gv("L", "R", prop_name="other")


def test_search_bidirectional_with_children():
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    df_n = pd.DataFrame({"node_id": ["A", "B", "C"], "layer": ["L", "L", "L"]})
    b.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {
            "source_id": ["A", "C"],
            "source_layer": ["L", "L"],
            "target_id": ["B", "B"],
            "target_layer": ["L", "L"],
        }
    )
    b.add_edges_from_dataframe(
        df_e, "source_id", "source_layer", "target_id", "target_layer", drop_na=False
    )
    s = OnionNetSearcher(core)
    gv = s.search(
        start_node_idx=1,
        max_dist=1,
        direction="bi",
        include_upstream_children=True,
        show_plot=False,
    )
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0, 1, 2}


def test_filter_edges_predicate_and_prune():
    core = make_simple_core()
    s = OnionNetSearcher(core)
    g = core.graph
    gv = s.filter_edges(lambda e: (g.ep["w"][e] % 2) == 1)
    for e in gv.edges():
        assert g.ep["w"][e] % 2 == 1
    for v in gv.vertices():
        assert (v.out_degree() + v.in_degree()) > 0
    epm = s.filter_edges(lambda e: True, return_view=False)
    assert hasattr(epm, "a") and len(epm.a) == g.num_edges() and epm.a.sum() == g.num_edges()


def test_search_plotting_paths_with_monkeypatch(monkeypatch):
    core = OnionNetGraph(directed=True)
    b = OnionNetBuilder(core)
    df_n = pd.DataFrame({"node_id": ["A", "B"], "layer": ["L", "L"]})
    b.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    df_e = pd.DataFrame(
        {"source_id": ["A"], "source_layer": ["L"], "target_id": ["B"], "target_layer": ["L"]}
    )
    b.add_edges_from_dataframe(
        df_e, "source_id", "source_layer", "target_id", "target_layer", drop_na=False
    )
    s = OnionNetSearcher(core)

    called = {"n": 0}

    def fake_draw(g, **kwargs):
        called["n"] += 1
        assert "vertex_text" in kwargs
        return None

    import onionnet.searcher as searcher_mod

    monkeypatch.setattr(searcher_mod, "graph_draw", fake_draw)

    s.search(start_node_idx=0, max_dist=1, direction="any", show_plot=True)
    from onionnet.property_manager import OnionNetPropertyManager

    OnionNetPropertyManager(core).create_node_label_property("node_label")
    s.search(start_node_idx=0, max_dist=1, direction="downstream", show_plot=True, verbosity=True)
    assert called["n"] >= 2
    gv_copy = s.view_layers(["L"], copy_gv=True)
    assert isinstance(gv_copy, gt.Graph)
    with pytest.raises(ValueError):
        s.search(start_node_idx=0, max_dist=1, direction="SIDEWAYS", show_plot=False)
