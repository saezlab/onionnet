import pandas as pd
import numpy as np
import pytest

from onionnet.builder import OnionNetBuilder
from onionnet.core import OnionNetGraph
from onionnet.property_manager import OnionNetPropertyManager


# --- Fixtures --------------------------------------------------------------

@pytest.fixture
def pm_and_graph_simple():
    # build a simple graph with one categorical v-prop and one categorical e-prop
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)

    # vertices A, B, C with categorical “grp”
    df_nodes = pd.DataFrame({
        "node_id": ["A", "B", "C"],
        "layer":   ["0", "0", "0"],
        "grp":     ["x", "y", "x"]
    })
    bldr.add_vertices_from_dataframe(df_nodes, "node_id", "layer",
                                     property_cols=["grp"], drop_na=False)

    # edges A→B and B→C with categorical “lbl”
    df_edges = pd.DataFrame({
        "source_id":    ["A", "B"],
        "source_layer": ["0", "0"],
        "target_id":    ["B", "C"],
        "target_layer": ["0", "0"],
        "lbl":          ["foo", "bar"]
    })
    bldr.add_edges_from_dataframe(df_edges,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["lbl"], drop_na=False)

    pm = OnionNetPropertyManager(core)
    return core, pm


@pytest.fixture
def pm_and_graph_mixed():
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)
    # seed some vertices with both numeric and categorical props
    df_n = pd.DataFrame({
        "node_id": ["A", "B", "C"],
        "layer":   ["0", "0", "0"],
        "grp":     ["x", "y", None],      # one None to test default mapping
        "score":   [0.5, np.nan, 1.5],    # one NaN to trigger astype(int) error
    })
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer",
                                     property_cols=["grp", "score"], drop_na=False)

    # seed some edges with categorical prop
    df_e = pd.DataFrame({
        "source_id":    ["A", "B", "C"],
        "source_layer": ["0", "0", "0"],
        "target_id":    ["B", "C", "A"],
        "target_layer": ["0", "0", "0"],
        "lbl":          ["foo", "bar", "baz"],
    })
    bldr.add_edges_from_dataframe(df_e,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["lbl"], drop_na=False)

    pm = OnionNetPropertyManager(core)
    return core, pm


# --- Tests for simple fixture ---------------------------------------------

def test_decode_vertex_property_labels_default(pm_and_graph_simple):
    core, pm = pm_and_graph_simple
    # decode the integer-coded "grp" into strings
    pm.decode_property_labels('v', 'grp')

    # check that a new vp "grp_decoded" exists
    assert 'grp_decoded' in core.graph.vp
    decoded = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    # both 'x' and 'y' must appear (be lenient in case a default sneaks in)
    assert {'x', 'y'}.issubset(set(decoded))


def test_decode_edge_property_labels_with_override(pm_and_graph_simple):
    core, pm = pm_and_graph_simple
    # supply a custom mapping: code 0→"X", code 1→"Y"; everything else →"Z"
    custom_map = {0: "X", 1: "Y"}
    pm.decode_property_labels(
        encoded_prop_type='e',
        encoded_prop_name='lbl',
        new_prop_name='human_lbl',
        mapping_dict=custom_map,
        default_label='Z'
    )

    assert 'human_lbl' in core.graph.ep
    vals = [core.graph.ep['human_lbl'][e] for e in core.graph.edges()]
    # original two codes 0,1 map to "X","Y"
    assert 'X' in vals and 'Y' in vals


@pytest.mark.parametrize("bad_dim", ['z', '', None])
def test_decode_property_labels_invalid_dim_raises(pm_and_graph_simple, bad_dim):
    _, pm = pm_and_graph_simple
    with pytest.raises(ValueError):
        pm.decode_property_labels(bad_dim, 'grp')


def test_decode_property_labels_missing_prop_raises(pm_and_graph_simple):
    _, pm = pm_and_graph_simple
    # no such vertex prop "nope"
    with pytest.raises(KeyError):
        pm.decode_property_labels('v', 'nope')
    # no such edge prop "nope"
    with pytest.raises(KeyError):
        pm.decode_property_labels('e', 'nope')


def test_decode_property_labels_bulk(pm_and_graph_simple, capsys):
    core, pm = pm_and_graph_simple
    # DataFrame with one object column and one numeric column
    df = pd.DataFrame({
        "grp":    ["x", "y", "x"],  # object → decoded
        "weight": [1.0, 2.0, 3.0]   # float → left alone
    })
    pm.decode_property_labels_bulk(df, encoded_prop_type='v')

    # object-type column should produce "grp_decoded"
    assert 'grp_decoded' in core.graph.vp

    # numeric column should be skipped with a printed notice
    out = capsys.readouterr().out
    assert "weight prop left as is" in out


# --- Tests for mixed fixture ----------------------------------------------

def test_decode_missing_codes_default_label(pm_and_graph_mixed):
    core, pm = pm_and_graph_mixed
    # override the mapping so that only code 0→"x" is known
    core.vertex_categorical_mappings["grp"]["int_to_str"] = {0: "x"}
    # everything else should map to "??"
    pm.decode_property_labels('v', 'grp', new_prop_name="grp2", default_label="??")
    decoded = [core.graph.vp["grp2"][v] for v in core.graph.vertices()]
    assert "x" in decoded
    assert "??" in decoded  # the None and the "y" both become "??"


def test_decode_astype_failure_raises(pm_and_graph_mixed):
    core, pm = pm_and_graph_mixed
    # score has a NaN → np.array(...).astype(int) will blow up (or at least be handled)
    with pytest.raises(ValueError):
        pm.decode_property_labels('v', 'score')


def test_bulk_decodes_edge_props_and_skips_numerics(pm_and_graph_mixed, capsys):
    core, pm = pm_and_graph_mixed
    # DataFrame with both object and numeric columns
    df = pd.DataFrame({
        "lbl":   ["foo", "bar", "baz"],
        "extra": [1, 2, 3]
    })
    pm.decode_property_labels_bulk(df, encoded_prop_type='e')
    # numeric “extra” should have been skipped with a message
    out = capsys.readouterr().out
    assert "extra prop left as is" in out

    # object “lbl” should become lbl_decoded on edges
    assert "lbl_decoded" in core.graph.ep
    vals = [core.graph.ep["lbl_decoded"][e] for e in core.graph.edges()]
    assert set(vals) == {"foo", "bar", "baz"}


def test_bulk_column_name_cleaning(pm_and_graph_simple):
    core, pm = pm_and_graph_simple
    # funky column names should get cleaned → new_prop_name uses cleaned
    df = pd.DataFrame({
        "My Col(1)/Test-Name": ["x", "y", "x"]
    })
    # first ensure we have a matching encoded prop
    core.vertex_categorical_mappings["My Col(1)/Test-Name"] = {
        "str_to_int": {"x": 0, "y": 1}, "int_to_str": {0: "x", 1: "y"}
    }
    # also create a dummy vp of int codes
    vp = core.graph.new_vertex_property("int")
    for v, code in zip(core.graph.vertices(), [0, 1, 0]):
        vp[v] = code
    core.graph.vp["My Col(1)/Test-Name"] = vp

    pm.decode_property_labels_bulk(df, encoded_prop_type='v')
    # cleaned “my_col1_test_name_decoded” should now be present
    assert "my_col1_test_name_decoded" in core.graph.vp