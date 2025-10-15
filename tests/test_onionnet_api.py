import graph_tool.all as gt
import pandas as pd

from onionnet.onionnet import OnionNet


def test_onionnet_wrappers_and_node_map():
    on = OnionNet(directed=True)
    # Build tiny graph
    df_nodes = pd.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "layer": ["L", "L", "R"],
            "lbl": ["a", "b", "c"],
        }
    )
    df_edges = pd.DataFrame(
        {
            "source_id": ["A", "B", "C"],
            "source_layer": ["L", "L", "R"],
            "target_id": ["B", "C", "A"],
            "target_layer": ["L", "L", "L"],
        }
    )
    on.grow_onion(
        df_nodes,
        df_edges,
        node_prop_cols=["lbl"],
        edge_prop_cols=None,
        drop_na=False,
        verbose=False,
    )

    # search wrapper (no plot)
    gv = on.search(start_node_idx=0, max_dist=1, direction="downstream", show_plot=False)
    assert isinstance(gv, gt.GraphView)

    # view_layers
    gvL = on.view_layers("L")
    assert isinstance(gvL, gt.GraphView)

    # view_components weak: whole graph likely connected
    comp = on.view_components(size_threshold=1, connectivity="weak")
    assert isinstance(comp, gt.GraphView)

    # filter_view_by_property on vertex label
    out = on.filter_view_by_property("lbl", {"a", "b"}, dim="v")
    assert isinstance(out, gt.GraphView)

    # compose_filters and create_bipartite_gv
    g = on.g

    def f_is_L(v, _g=g, _code=on.core.layer_name_to_code["L"]):
        return _g.vp["layer_hash"][v] == _code

    cf = on.compose_filters([f_is_L], type="v")
    assert isinstance(cf, gt.GraphView)
    bi = on.create_bipartite_gv("L", "R")
    assert isinstance(bi, gt.GraphView)

    # filter_edges_between_categories wrapper
    gv_lr = on.filter_edges_between_categories("L", "R", mode="both")
    assert isinstance(gv_lr, gt.GraphView)

    # node_map caches and contains entries
    nm1 = on.node_map
    nm2 = on.node_map  # cached
    assert nm1 is nm2 and ("L", "A") in nm1

    # property manager wrappers
    # Name tuple
    v = on.get_vertex_by_name_tuple("L", "A")
    assert int(v) == nm1[("L", "A")]
    # Encoding tuple
    layer_code = on.core.layer_name_to_code["L"]
    node_int = on.core.node_id_str_to_int["A"]
    v2 = on.get_vertex_by_encoding_tuple(layer_code, node_int)
    assert int(v2) == int(v)
    # get/set property
    on.set_vertex_property(layer_code, node_int, "score", 7)
    assert on.get_vertex_property(layer_code, node_int, "score") == 7
    # view properties
    props_all = on.view_node_properties(layer_code, node_int)
    assert (
        isinstance(props_all, dict)
        and "decoded_layer" in props_all
        and "decoded_node_id" in props_all
    )
    props_some = on.view_node_properties_by_names("L", "A", verbose=True)
    assert isinstance(props_some, dict)
    # create labels and bulk decode
    on.create_node_label_property("node_label")
    df_props = pd.DataFrame({"lbl": ["a"]})
    on.decode_property_labels_bulk(df_props, encoded_prop_type="v")
