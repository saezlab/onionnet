import pytest
import numpy as np
from onionnet.core import OnionNetGraph
from onionnet.builder import OnionNetBuilder
from onionnet.searcher import OnionNetSearcher
from graph_tool.all import GraphView, shortest_distance
import pandas as pd


# --- Fixtures --------------------------------------------------------------

@pytest.fixture
def simple_graph():
    """Tiny 2-node, 1-edge graph with a numeric vertex prop 'score'."""
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    df = pd.DataFrame({
        "node_id": ["A","B"],
        "layer":   ["0","0"],
        "score":   [10,20]
    })
    b.add_vertices_from_dataframe(df, "node_id","layer", property_cols=["score"], drop_na=False)
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
    df_nodes = pytest.importorskip("pandas").DataFrame({
        "node_id": ["A","B","C"],
        "layer":   ["0","0","0"]
    })
    b.add_vertices_from_dataframe(df_nodes, "node_id","layer", drop_na=False)
    df_edges = pytest.importorskip("pandas").DataFrame({
        "source_id":    ["A","B"],
        "source_layer": ["0","0"],
        "target_id":    ["B","C"],
        "target_layer": ["0","0"],
    })
    b.add_edges_from_dataframe(df_edges,
        "source_id","source_layer",
        "target_id","target_layer",
        property_cols=None, drop_na=False)
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
    assert gv.num_edges()    == 2


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
    gv = s.search(start_node_idx=0, max_dist=1, direction='downstream', show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0,1}


def test_search_upstream(chain_graph_and_searcher):
    """
    Upstream search from C with max_dist=1 should include C and B only.
    """
    core, s = chain_graph_and_searcher
    gv = s.search(start_node_idx=2, max_dist=1, direction='upstream', show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {1,2}


def test_search_bidirectional(chain_graph_and_searcher):
    """
    Bidirectional search from B max_dist=1 should include A,B,C.
    """
    core, s = chain_graph_and_searcher
    gv = s.search(start_node_idx=1, max_dist=1, direction='bi', show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0,1,2}


def test_search_any(chain_graph_and_searcher):
    """
    'any' search treats the graph as undirected.
    From C with max_dist=2 in an undirected chain, should reach A,B,C.
    """
    core, s = chain_graph_and_searcher
    gv = s.search(start_node_idx=2, max_dist=2, direction='any', show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0,1,2}


def test_bi_vs_any_distinction(builder_and_core):
    """
    In this “V-shape” graph (A→B and C→B), a bidirectional ('bi') search
    from A with max_dist=2 will only see A→B (so {A,B}), because there is no
    directed path A→B→C.  But an undirected ('any') search at max_dist=2
    will go A–B–C and thus pick up C as well.
    """
    bldr, core = builder_and_core

    # 1) Add our three nodes A, B, C all on layer "0"
    df_n = pd.DataFrame({
        "node_id": ["A", "B", "C"],
        "layer":   ["0", "0", "0"]
    })
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)

    # 2) Add two directed edges: A→B and C→B
    df_e = pd.DataFrame({
        "source_id":    ["A",    "C"],
        "source_layer": ["0",    "0"],
        "target_id":    ["B",    "B"],
        "target_layer": ["0",    "0"],
    })
    bldr.add_edges_from_dataframe(
        df_e,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=None,
        drop_na=False
    )

    s = OnionNetSearcher(core)

    # BIDIRECTIONAL: union of upstream(from A) and downstream(from A)
    # downstream(A) = {A,B}, upstream(A) = {A}  →  {A,B}
    gv_bi = s.search(start_node_idx=0, max_dist=2, direction='bi', show_plot=False)
    got_bi = {int(v) for v in gv_bi.vertices()}
    assert got_bi == {0, 1}, f"expected just A,B for 'bi', got {got_bi}"

    # ANY (undirected): can traverse A–B–C in two hops → {A,B,C}
    gv_any = s.search(start_node_idx=0, max_dist=2, direction='any', show_plot=False)
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
    assert hasattr(vfilt, 'get_array')


def test_view_layers_invalid(chain_graph_and_searcher):
    """
    Asking for a non-existent layer should ValueError.
    """
    _, s = chain_graph_and_searcher
    with pytest.raises(ValueError):
        s.view_layers("nonexistent")


@pytest.mark.parametrize("connectivity, expected_size", [
    ("strong", 0),  # no strongly‐connected components of size ≥ 2 in a simple chain
    ("weak",   3),  # undirected (weak) connectivity groups the whole chain into one component
])
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
        prop_name="score",
        target_value=15,
        comparison=">",
        dim='v'
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
        prop_name="w",
        target_value=[10],
        dim='e',
        prune_isolated=True
    )

    # Only one edge (the one with weight=10) remains,
    # and its two incident nodes must both survive.
    assert gv.num_edges()    == 1
    assert gv.num_vertices() == 2


def test_compose_filters_and_return_prop(chain_graph_and_searcher):
    """
    compose_filters can combine predicates and return a PropertyMap.
    """
    core, s = chain_graph_and_searcher
    # predicate: nodes A or C only
    funcs = [lambda v: int(v) == 0, lambda v: int(v) == 2]
    prop = s.compose_filters(funcs, mode="or", type='v', return_prop=True)
    arr = [bool(prop[v]) for v in core.graph.vertices()]
    assert arr == [True, False, True]
    # as GraphView
    gv = s.compose_filters(funcs, mode="and", type='v', return_prop=False)
    assert isinstance(gv, GraphView)


def test_create_bipartite_gv(builder_and_core):
    """
    create_bipartite_gv should only keep edges between two specified layers.
    """
    core = OnionNetGraph()
    b = OnionNetBuilder(core)
    df = pytest.importorskip("pandas").DataFrame({
        "node_id": ["A","B","C","D"],
        "layer":   ["L1","L2","L1","L2"]
    })
    b.add_vertices_from_dataframe(df, "node_id","layer", property_cols=None, drop_na=False)
    # encode 'layer' as string vp
    lbl = core.graph.new_vertex_property("string")
    for v in core.graph.vertices():
        lbl[v] = core.layer_code_to_name[ core.graph.vp['layer_hash'][v] ]
    core.graph.vp["layer_decoded"] = lbl

    # edges crossing and within layers
    df_e = pytest.importorskip("pandas").DataFrame({
        "source_id":    ["A","A","C"],
        "source_layer": ["L1","L1","L1"],
        "target_id":    ["B","C","B"],
        "target_layer": ["L2","L1","L2"]
    })
    b.add_edges_from_dataframe(df_e,"source_id","source_layer","target_id","target_layer",
                               property_cols=None, drop_na=False)
    s = OnionNetSearcher(core)
    gv = s.create_bipartite_gv("L1","L2","layer_decoded")
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
        s.search(start_node_idx=0, max_dist=1, direction='sideways', show_plot=False)


def test_filter_view_by_property_invalid_prop(simple_graph):
    """
    filter_view_by_property should raise ValueError when asked
    for a non-existent property (both vertex and edge) or invalid comparison.
    """
    core = simple_graph
    s = OnionNetSearcher(core)
    # vertex side invalid prop
    with pytest.raises(ValueError):
        s.filter_view_by_property("does_not_exist", 1, dim='v')
    # edge side invalid prop
    with pytest.raises(ValueError):
        s.filter_view_by_property("does_not_exist", 1, dim='e')
    # invalid comparison operator
    with pytest.raises(ValueError):
        s.filter_view_by_property("score", 1, comparison="<>", dim='v')


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
    view_ab = GraphView(core.graph, vfilt=lambda v: int(v) in {0,1})
    gv = s.search(start_node_idx=0, max_dist=2, direction='any', g=view_ab, show_plot=False)
    vs = {int(v) for v in gv.vertices()}
    assert vs == {0,1}


@pytest.mark.parametrize("mode", ["xor", "not"])
def test_compose_filters_bad_mode(chain_graph_and_searcher, mode):
    """
    compose_filters should reject unsupported combination modes
    and raise ValueError.
    """
    _, s = chain_graph_and_searcher
    with pytest.raises(ValueError):
        s.compose_filters([lambda v: True], mode=mode, type='v')


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
    gv = s.filter_view_by_property("w", 999, dim='e', prune_isolated=False)
    assert gv.num_edges() == 0
    assert gv.num_vertices() == core.graph.num_vertices()


def test_create_bipartite_gv_no_cross_edges(builder_and_core): # TODO: check if semantically correct
    """
    create_bipartite_gv on a graph with two layers but no edges
    should return an empty GraphView (0 vertices, 0 edges).
    """
    core = OnionNetGraph(); b = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A","B"], "layer": ["X","Y"]})
    b.add_vertices_from_dataframe(df, "node_id","layer", drop_na=False)
    s = OnionNetSearcher(core)
    gv = s.create_bipartite_gv("X","Y","layer_decoded") # in practice would use layer_decoded
    assert gv.num_vertices() == 0 and gv.num_edges() == 0


def test_create_bipartite_gv_invalid_prop(builder_and_core): # TODO: check if semantically correct
    """
    create_bipartite_gv should raise KeyError when asked
    to use a non-existent vertex property for filtering.
    """
    core = OnionNetGraph(); b = OnionNetBuilder(core)
    df = pd.DataFrame({"node_id": ["A","B"], "layer": ["X","Y"]})
    b.add_vertices_from_dataframe(df, "node_id","layer", drop_na=False)
    s = OnionNetSearcher(core)
    with pytest.raises(KeyError):
        s.create_bipartite_gv("X","Y","does_not_exist")
