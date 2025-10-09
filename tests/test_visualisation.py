import os
import tempfile

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')      # <-- switch to headless backend
import graph_tool.all as gt
from graph_tool.all import Graph, GraphView, sfdp_layout

from onionnet.visualisation import (
    flatten_properties,
    create_node_labels,
    color_nodes,
    shape_nodes,
    add_halo_to_node,
    add_halos_to_nodes,
    set_node_sizes_and_text_by_depth,
    get_legend,
    color_edges,
    layout_by_layer,
    bipartite_ordered_layout,
    load_or_compute_layout,
    prop_to_size,
)
from onionnet.core import OnionNetGraph


# --- flatten_properties & create_node_labels --------------------------------

def test_flatten_properties_mixed_nested():
    """
    flatten_properties should recursively unpack nested lists and stringify.
    """
    nested = ["a", ["b", ["c", 1], "a"], 2]
    flat = flatten_properties(nested)
    # order matters (depth-first), duplicates preserved
    assert flat == ["a", "b", "c", "1", "a", "2"]


def test_create_node_labels_removes_duplicates_and_handles_empty():
    """
    create_node_labels should flatten+dedupe per-vertex and handle empty entries.
    """
    g = gt.Graph(directed=False)
    g.add_vertex(3)
    # property map of lists
    pm = g.new_vertex_property("object")
    pm[g.vertex(0)] = ["x", ["y", "x"]]
    pm[g.vertex(1)] = []
    pm[g.vertex(2)] = None
    g.vp["nested"] = pm

    labels = create_node_labels(g, g.vp["nested"])
    # vertex 0: {x,y}
    lbl0 = set(labels[g.vertex(0)].split(", "))
    assert lbl0 == {"x", "y"}
    # vertex 1 & 2: empty string
    assert labels[g.vertex(1)] == ""
    assert labels[g.vertex(2)] == ""


# --- color_nodes -------------------------------------------------------------

@pytest.fixture
def cat_graph():
    """
    Simple 3-node graph with a categorical prop 'grp' and numeric prop 'val'.
    """
    g = gt.Graph(directed=False)
    v0, v1, v2 = g.add_vertex(), g.add_vertex(), g.add_vertex()
    pm_cat = g.new_vertex_property("string")
    pm_val = g.new_vertex_property("double")
    for i, v in enumerate(g.vertices()):
        pm_cat[v] = ["A","B","A"][int(v)]
        pm_val[v] = float([0.0, 0.5, 1.0][int(v)])
    g.vp["grp"] = pm_cat
    g.vp["val"] = pm_val
    return g

def test_color_nodes_categorical_and_legend(cat_graph):
    """
    color_nodes categorical: correct RGBA, legend keys = categories.
    """
    res = color_nodes(cat_graph, "grp", method="categorical", generate_legend=True, transparency=0.8)
    vcol = res["v_color"]
    legend = res["legend_node_color"]
    # All RGBA vectors length 4
    for v in cat_graph.vertices():
        assert len(vcol[v]) == 4 and pytest.approx(vcol[v][3]) == 0.8
    # Legend covers exactly {'A','B'}
    assert set(legend) == {"A","B"}


def test_color_nodes_continuous_zero_centred_and_legend(cat_graph):
    """
    color_nodes continuous: zero-centred normalization and legend contents.
    """
    res = color_nodes(cat_graph, "val", method="continuous", zero_centred=True, generate_legend=True)
    legend = res["legend_node_color"]
    # legend must include min_val==-max_val symmetry
    assert pytest.approx(legend["min_val"] + legend["max_val"]) == 0.0
    # v_color respects range
    vcol = res["v_color"]
    vals = [float(cat_graph.vp["val"][v]) for v in cat_graph.vertices()]
    for v in cat_graph.vertices():
        assert len(vcol[v]) == 4


def test_color_nodes_boolean_and_custom_dict(cat_graph):
    """
    color_nodes boolean: True->red, False->grey. custom_color_dict override works or errors.
    """
    # create a bool prop
    pm_bool = cat_graph.new_vertex_property("bool")
    for i, v in enumerate(cat_graph.vertices()):
        pm_bool[v] = (i % 2 == 0)
    cat_graph.vp["flag"] = pm_bool

    res_bool = color_nodes(cat_graph, "flag", method="boolean", generate_legend=True)
    legend = res_bool["legend_node_color"]
    assert legend == {"True": (1.0,0.0,0.0,1.0), "False": (0.5,0.5,0.5,1.0)}

    # custom_color_dict covers only True, missing False should raise
    with pytest.raises(ValueError):
        color_nodes(cat_graph, "flag", custom_color_dict={True:(0,1,0)}, generate_legend=False)


# --- shape_nodes -------------------------------------------------------------

@pytest.fixture
def shape_graph():
    g = gt.Graph(directed=False)
    for _ in range(3): g.add_vertex()
    pm = g.new_vertex_property("string")
    pm[g.vertex(0)] = "X"
    pm[g.vertex(1)] = "Y"
    pm[g.vertex(2)] = "X"
    g.vp["cat"] = pm
    return g

def test_shape_nodes_categorical_and_boolean(shape_graph):
    """
    shape_nodes categorical and boolean mode + legends.
    """
    res_cat = shape_nodes(shape_graph, "cat", shape_method="categorical", generate_legend=True)
    leg_cat = res_cat["legend_node_shape"]
    assert set(leg_cat) == set(shape_graph.vp["cat"][v] for v in shape_graph.vertices())

    # boolean on an existing bool prop
    pmb = shape_graph.new_vertex_property("bool")
    for v in shape_graph.vertices(): pmb[v] = (int(v) % 2 == 0)
    shape_graph.vp["b"] = pmb
    res_bool = shape_nodes(shape_graph, "b", shape_method="boolean", generate_legend=True)
    assert res_bool["legend_node_shape"] == {"True":"triangle","False":"square"}

    # custom_shape_dict missing key -> error
    with pytest.raises(ValueError):
        shape_nodes(shape_graph, "cat", custom_shape_dict={"Z":"hexagon"})


# --- add_halo_to_node -------------------------------------------------------

def test_add_halo_to_node_single(shape_graph):
    """
    add_halo_to_node should mark exactly the chosen node.
    """
    g = shape_graph
    target = g.vertex(1)
    res = add_halo_to_node(g, target, halo_color=(0.1,0.2,0.3,0.4))
    halo_map = res["v_halo"]
    halo_col = res["v_halo_color"]
    for v in g.vertices():
        if v == target:
            assert halo_map[v]
            assert tuple(halo_col[v]) == (0.1,0.2,0.3,0.4)
        else:
            assert not halo_map[v]


# --- set_node_sizes_and_text_by_depth ---------------------------------------

def test_set_node_sizes_and_text_by_depth_chain():
    """
    On a 3-node chain, depth scaling gives decreasing sizes.
    """
    g = gt.Graph(directed=False)
    a,b,c = g.add_vertex(), g.add_vertex(), g.add_vertex()
    g.add_edge(a,b); g.add_edge(b,c)
    v_size, v_text = set_node_sizes_and_text_by_depth(g, root=a, max_size=10, min_size=4, max_text_size=12, min_text_size=6)
    # root depth=0 => largest size
    assert v_size[a] > v_size[b] > v_size[c]
    assert v_text[a] > v_text[b] > v_text[c]


# --- get_legend --------------------------------------------------------------

def test_get_legend_from_dicts_and_graph(tmp_path, capsys, cat_graph):
    """
    get_legend handles:
      - categorical dict
      - continuous dict missing keys → error
      - graph categorical & continuous
    """
    # 1) categorical dict → no error
    cat_dict = {"A": (1, 0, 0, 1), "B": (0, 1, 0, 1)}
    get_legend(cat_dict, title="Test")

    # 2) continuous dict missing min_val → ValueError
    with pytest.raises(ValueError):
        get_legend({"min_col": (0,0,0,1), "max_col": (1,1,1,1)}, prop="val")

    # 3) graph categorical mode → no error
    get_legend(cat_graph, prop="grp", verbose=True)

    # 4) graph continuous mode needs a plotted image in an Axes first:
    pmn = cat_graph.new_vertex_property("double")
    for i, v in enumerate(cat_graph.vertices()):
        pmn[v] = float(i)
    cat_graph.vp["num"] = pmn

    # Create a headless figure/axes and draw a dummy image so colorbar() has
    # an “artist” to attach to.
    fig, ax = plt.subplots()
    # Dummy 2×1 image spanning the full colormap range
    data = np.linspace(0, 1, 2).reshape(1, 2)
    ax.imshow(data, cmap='viridis', norm=plt.Normalize(0, float(len(list(cat_graph.vertices())))-1))
    # Now get_legend can successfully call plt.colorbar()
    get_legend(cat_graph, prop="num", mode="continuous")
    plt.close(fig)


# --- color_edges ------------------------------------------------------------

@pytest.fixture
def edge_graph():
    g = gt.Graph(directed=False)
    for _ in range(3): g.add_vertex()
    e0 = g.add_edge(g.vertex(0), g.vertex(1))
    e1 = g.add_edge(g.vertex(1), g.vertex(2))
    pm = g.new_edge_property("int")
    pm[e0], pm[e1] = 5, 10
    g.ep["w"] = pm
    return g

def test_color_edges_categorical_and_continuous(edge_graph):
    """
    color_edges should mirror color_nodes behavior for edges.
    """
    # categorical
    res_cat = color_edges(edge_graph, "w", method="categorical", generate_legend=True)
    assert set(res_cat["legend_edge_color"].keys()) == {5,10}
    # continuous zero-centred
    res_cont = color_edges(edge_graph, "w", method="continuous", zero_centred=True, generate_legend=True)
    assert "min_val" in res_cont["legend_edge_color"]


# --- layout_by_layer --------------------------------------------------------

@pytest.fixture
def layered_graph():
    g = gt.Graph(directed=False)
    v0 = g.add_vertex(); v1 = g.add_vertex()
    pm = g.new_vertex_property("string")
    pm[v0], pm[v1] = "L1","L2"
    g.vp["layer_decoded"] = pm
    return g

def test_layout_by_layer_two_layers(layered_graph):
    """
    layout_by_layer should place two layers horizontally apart.
    """
    pos = layout_by_layer(layered_graph, layer_prop_name="layer_decoded", spacing=10, epsilon=0.1)
    xs = [pos[v][0] for v in layered_graph.vertices()]
    assert max(xs)-min(xs) >= 10

def test_layout_by_layer_degenerate_raises(layered_graph):
    """
    If only one layer has one node, width=0 → error.
    """
    # give both nodes same layer => width=0
    layered_graph.vp["layer_decoded"][layered_graph.vertex(1)] = "L1"
    with pytest.raises(ValueError):
        layout_by_layer(layered_graph, layer_prop_name="layer_decoded", spacing=10, epsilon=0.01)


# --- bipartite_ordered_layout ------------------------------------------------

@pytest.fixture
def bipartite_graph():
    g = gt.Graph(directed=False)
    # left=0,1; right=2,3
    for _ in range(4): g.add_vertex()
    pm = g.new_vertex_property("string")
    pm[g.vertex(0)], pm[g.vertex(1)] = "L","L"
    pm[g.vertex(2)], pm[g.vertex(3)] = "R","R"
    g.vp["layer_decoded"] = pm
    # connect 0->2,1->3
    g.add_edge(g.vertex(0), g.vertex(2))
    g.add_edge(g.vertex(1), g.vertex(3))
    return g

def test_bipartite_ordered_layout_positions(bipartite_graph):
    """
    bipartite_ordered_layout should set x=0 for L, x=1 for R, y in sorted order.
    """
    pos = bipartite_ordered_layout(bipartite_graph, left_val="L", right_val="R", horizontal_spacing=2.0, vertical_spacing=5.0)
    for v in bipartite_graph.vertices():
        layer = bipartite_graph.vp["layer_decoded"][v]
        x,y = pos[v]
        if layer == "L":
            assert pytest.approx(x)==0.0
        else:
            assert pytest.approx(x)==2.0


# --- load_or_compute_layout --------------------------------------------------

def test_load_or_compute_layout_inject_and_load(tmp_path):
    """
    load_or_compute_layout with inject builds and saves TSV, then reload reads it back.
    """
    core = OnionNetGraph()
    b = gt.GraphView  # dummy to import
    # build a 2-node OnionNetGraph
    bldr = __import__('onionnet.builder').builder.OnionNetBuilder(core)
    df = pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]})
    bldr.add_vertices_from_dataframe(df,"node_id","layer",drop_na=False)
    g = core.graph

    fn = tmp_path/"layout.tsv"
    # inject: give constant layout
    inject = lambda G: (G.new_vertex_property("vector<double>"), None)[0] and {v:[1.0,2.0] for v in G.vertices()}
    # actually need a propmap: let's compute real SFDP
    pos1 = load_or_compute_layout(g, str(fn), override=False, inject=lambda G: {v:[0.1*i,0.2*i] for i,v in enumerate(G.vertices())})
    assert fn.exists()
    # now load from file
    pos2 = load_or_compute_layout(g, str(fn), override=False, inject=None)
    # positions agree
    for v in g.vertices():
        assert pytest.approx(pos1[v]) == pos2[v]

def test_load_or_compute_layout_bad_graph(tmp_path):
    """
    load_or_compute_layout on bare Graph without keys → ValueError.
    """
    g = gt.Graph()
    with pytest.raises(ValueError):
        load_or_compute_layout(g, str(tmp_path/"x.tsv"), override=False, inject=None)


# --- prop_to_size ------------------------------------------------------------

def test_prop_to_size_linear_and_power():
    """
    prop_to_size should map values to [mi,ma], respecting power and mode.
    """
    arr = [0, 5, 10]
    g = gt.Graph(directed=False)
    for _ in arr: g.add_vertex()
    # as vertex prop
    vp = prop_to_size(g, arr, mi=1, ma=11, power=1, mode='v')
    vals = [vp[v] for v in g.vertices()]
    assert vals == [1, 6, 11]

    # with square power: both middle and top get clamped to ma=4
    vp2 = prop_to_size(g, arr, mi=0, ma=4, power=2, mode='v')
    # now they should be equal (both hit the max)
    assert vp2[1] == vp2[2] == 4


def test_prop_to_size_constant_and_edge_mode(edge_graph):
    """
    prop_to_size constant values => all mi; edge mode produces an edge map.
    """
    sizes = [5,5,5]
    g = edge_graph
    ep = prop_to_size(g, sizes, mi=2, ma=8, mode='e')
    for e in g.edges():
        assert ep[e] == 2
    # invalid mode
    with pytest.raises(ValueError):
        prop_to_size(g, sizes, mode='x')


# ---- Additional visualisation tests ----


def test_add_halos_to_nodes_on_graph_and_view():
    g = gt.Graph(directed=False)
    g.add_vertex(4)
    vf = g.new_vertex_property('bool'); vf.a = np.array([False, True, True, True])
    gv = gt.GraphView(g, vfilt=vf)
    out = add_halos_to_nodes(gv, nodes=[1, 3], colors=[(0,1,0,0.8)])
    v_halo = out['v_halo']; v_col = out['v_halo_color']
    assert 'v_halo' in gv.vp and 'v_halo_color' in gv.vp
    assert v_halo[g.vertex(1)] and v_halo[g.vertex(3)] and not v_halo[g.vertex(2)]
    assert tuple(v_col[g.vertex(1)]) == (0,1,0,0.8)
    assert tuple(v_col[g.vertex(3)]) != (0,1,0,0.8)
    out2 = add_halos_to_nodes(g, nodes=[0,2])
    assert 'v_halo' in g.vp and out2['v_halo'][g.vertex(0)]


def test_get_legend_dict_continuous_and_shapes():
    dcont = {'min_col': (0,0,0,1), 'max_col': (1,1,1,1), 'min_val': 0.0, 'max_val': 10.0}
    get_legend(dcont, prop='score', mode='continuous')
    dshape = {'A': 'circle', 'B': 'triangle', 'C': 'square'}
    get_legend(dshape, title='Shapes')


def test_get_legend_graph_categorical_and_save(tmp_path):
    g = gt.Graph(directed=False)
    g.add_vertex(3)
    cat = g.new_vertex_property('string')
    for i, v in enumerate(g.vertices()):
        cat[v] = ['x','y','x'][i]
    g.vp['grp'] = cat
    fn = tmp_path/"legend_svg"
    get_legend(g, prop='grp', mode='categorical', ordered_cats=['y','x'], save_filename=str(fn))
    assert (tmp_path/"legend_svg.svg").exists()


def test_load_or_compute_layout_decoded_keys(tmp_path):
    g = gt.Graph(directed=False)
    g.add_vertex(2)
    ld = g.new_vertex_property('string'); nd = g.new_vertex_property('string')
    for i, v in enumerate(g.vertices()):
        ld[v] = ['L','R'][i]; nd[v] = ['A','B'][i]
    g.vp['layer_decoded'] = ld
    g.vp['node_id_decoded'] = nd
    fn = tmp_path/"dec_layout.tsv"
    pos = load_or_compute_layout(g, str(fn), override=True)
    assert fn.exists()
    df = pd.read_csv(fn, sep='\t')
    assert set(['layer_decoded','node_id_decoded','x','y']).issubset(df.columns)
    pos2 = load_or_compute_layout(g, str(fn), override=False)
    for v in g.vertices():
        assert pytest.approx(pos[v][0]) == pos2[v][0]


def test_get_legend_graph_requires_prop_raises():
    g = gt.Graph()
    g.add_vertex(1)
    with pytest.raises(ValueError):
        get_legend(g)


def test_layout_by_layer_missing_prop_raises():
    g = gt.Graph(directed=False)
    g.add_vertex(1)
    with pytest.raises(KeyError):
        layout_by_layer(g, layer_prop_name='layer_decoded')


#### More comprehensive tests for layouts


@pytest.fixture
def tmp_tsv(tmp_path):
    return str(tmp_path / "layout.tsv")

def make_simple_graph(v_count=3, directed=False):
    """
    Helper: builds a Graph with v_count vertices and sets up either:
    - layer_hash & node_id_hash (default), or
    - v_int only, if you pop off the hashes.
    """
    g = Graph(directed=directed)
    for _ in range(v_count):
        g.add_vertex()
    # simulate builder: fill in layer_hash & node_id_hash + v_int
    g.vp["layer_hash"]   = g.new_vertex_property("int")
    g.vp["node_id_hash"] = g.new_vertex_property("int")
    g.vp["v_int"]        = g.new_vertex_property("int")
    for i, v in enumerate(g.vertices()):
        g.vp["layer_hash"][v]   = 0
        g.vp["node_id_hash"][v] = i
        g.vp["v_int"][v]        = i
    return g

def test_missing_key_properties_raises(tmp_tsv):
    # Graph with neither hash nor v_int → error
    g = Graph()
    with pytest.raises(ValueError):
        load_or_compute_layout(g, tmp_tsv)

def test_inject_callable_and_file_written(tmp_tsv):
    # If inject is a callable, should use it, write file, and return its layout
    g = make_simple_graph(4)
    def inject_fn(graph):
        prop = graph.new_vertex_property("vector<double>")
        for v in graph.vertices():
            prop[v] = [int(v), int(v)*2]
        return prop

    pos = load_or_compute_layout(g, tmp_tsv, inject=inject_fn)
    # ensure returned prop matches our inject
    coords = [tuple(pos[v]) for v in g.vertices()]
    assert coords == [(i, i*2) for i in range(4)]

    # file must exist with matching columns and rows
    df = pd.read_csv(tmp_tsv, sep="\t")
    assert set(df.columns) == {"layer_hash","node_id_hash","x","y"}
    assert len(df) == 4

def test_inject_array_and_file_written(tmp_tsv):
    # If inject is precomputed mapping (not callable)
    g = make_simple_graph(2)
    pre = g.new_vertex_property("vector<double>")
    pre[g.vertex(0)] = [0.1, 0.2]
    pre[g.vertex(1)] = [1.1, 1.2]

    pos = load_or_compute_layout(g, tmp_tsv, inject=pre)
    assert tuple(pos[g.vertex(1)]) == (1.1, 1.2)

    df = pd.read_csv(tmp_tsv, sep="\t")
    assert np.allclose(df["x"].values, [0.1,1.1])
    assert np.allclose(df["y"].values, [0.2,1.2])

def test_compute_and_load_vs_override(tmp_tsv):
    # First run writes via sfdp_layout
    g = make_simple_graph(3)
    pos1 = load_or_compute_layout(g, tmp_tsv, override=False, inject=None)
    df1 = pd.read_csv(tmp_tsv, sep="\t")
    # Second run without override should load the same positions
    pos2 = load_or_compute_layout(g, tmp_tsv, override=False, inject=None)
    coords1 = [tuple(pos1[v]) for v in g.vertices()]
    coords2 = [tuple(pos2[v]) for v in g.vertices()]
    assert coords1 == coords2

    # Now force override: new sfdp_layout (almost always different)
    pos3 = load_or_compute_layout(g, tmp_tsv, override=True, inject=None)
    df3 = pd.read_csv(tmp_tsv, sep="\t")
    coords3 = [tuple(pos3[v]) for v in g.vertices()]
    # file updated with override
    assert len(df3) == 3
    # very likely at least one coordinate differs
    assert coords3 != coords1

def test_missing_rows_in_file_error(tmp_tsv):
    # Write an incomplete file missing one vertex
    g = make_simple_graph(3)
    df = pd.DataFrame({
        "layer_hash": [0,0],
        "node_id_hash":[0,1],
        "x":[0,1],
        "y":[0,1]
    })
    df.to_csv(tmp_tsv, sep="\t", index=False)
    # Loading should complain about missing layout for v_int 2
    with pytest.raises(ValueError):
        load_or_compute_layout(g, tmp_tsv, override=False)

def test_v_int_only_branch(tmp_tsv):
    # Remove hash props so only v_int remains
    g = make_simple_graph(3)
    del g.vp["layer_hash"]
    del g.vp["node_id_hash"]

    # Compute → writes v_int key
    pos = load_or_compute_layout(g, tmp_tsv)
    df = pd.read_csv(tmp_tsv, sep="\t")
    assert "v_int" in df.columns and "x" in df.columns and "y" in df.columns

    # Edit file to drop the required column → should error
    df = df.drop(columns=["v_int"])
    df.to_csv(tmp_tsv, sep="\t", index=False)
    with pytest.raises(ValueError):
        load_or_compute_layout(g, tmp_tsv, override=False)
