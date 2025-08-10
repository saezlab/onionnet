import pandas as pd
import numpy as np
import pytest

from onionnet.builder import OnionNetBuilder
from onionnet.core import OnionNetGraph
from onionnet.property_manager import OnionNetPropertyManager


# --- Fixtures for vertex access/manipulation tests --------------------

@pytest.fixture
def access_core_pm():
    """
    Simple graph with two vertices ('A' on layer 'L1', 'B' on layer 'L2') and a numeric vertex prop 'p'.
    Used to test get/set/view routines independently of the more complex categorical fixtures.
    """
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)
    df = pd.DataFrame({
        "node_id": ["A", "B"],
        "layer":   ["L1", "L2"],
        "p":       [1, 2]
    })
    bldr.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=["p"], drop_na=False)
    pm = OnionNetPropertyManager(core)
    return core, pm


# --- Fixtures for more complex decoding --------------------------------

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

#### SIMPLE FUNCTIONS

# --- Tests for identifier lookups -----------------------------------------

def test_get_vertex_by_encoding_and_name_lookup_success(access_core_pm):
    # Ensure that looking up a vertex by its encoded tuple and by its human-readable (name) tuple returns the same vertex.
    core, pm = access_core_pm
    layer_code = core._map_layer("L1")
    node_id_int = core._map_node_id("A")
    v_enc = pm.get_vertex_by_encoding_tuple(layer_code, node_id_int)
    v_name = pm.get_vertex_by_name_tuple("L1", "A")
    assert v_enc is not None
    assert v_enc == v_name

def test_get_vertex_by_encoding_tuple_not_found_returns_none(access_core_pm):
    # Missing encoding tuple should yield None rather than blowing up.
    core, pm = access_core_pm
    v = pm.get_vertex_by_encoding_tuple(9999, 8888)  # nonexistent layer/node combo
    assert v is None

def test_get_vertex_by_name_tuple_missing_raises(access_core_pm):
    # Asking for a vertex by a name tuple with unknown layer or node_id should raise KeyError.
    _, pm = access_core_pm
    with pytest.raises(KeyError):
        pm.get_vertex_by_name_tuple("NONEXISTENT", "A")
    with pytest.raises(KeyError):
        pm.get_vertex_by_name_tuple("L1", "Z")


# --- Tests for property access & mutation ----------------------------------

def test_get_vertex_property_existing_and_missing(access_core_pm):
    # Existing property should be returned; missing property or missing vertex returns None.
    core, pm = access_core_pm
    layer_code = core._map_layer("L1")
    node_id_int = core._map_node_id("A")
    # existing prop
    assert pm.get_vertex_property(layer_code, node_id_int, "p") == 1
    # nonexistent property
    assert pm.get_vertex_property(layer_code, node_id_int, "does_not_exist") is None
    # nonexistent vertex
    assert pm.get_vertex_property(999, 999, "p") is None

def test_set_vertex_property_creates_and_updates(access_core_pm):
    # Setting a new property on an existing vertex creates it and updates value; subsequent sets overwrite.
    core, pm = access_core_pm
    layer_code = core._map_layer("L1")
    node_id_int = core._map_node_id("A")
    # create new property 'q'
    pm.set_vertex_property(layer_code, node_id_int, "q", 42)
    assert "q" in core.graph.vp
    assert pm.get_vertex_property(layer_code, node_id_int, "q") == 42
    # update it
    pm.set_vertex_property(layer_code, node_id_int, "q", 99)
    assert pm.get_vertex_property(layer_code, node_id_int, "q") == 99

def test_set_vertex_property_on_missing_vertex_prints(access_core_pm, capsys):
    # Attempting to set a property on a missing vertex should not raise, but should print an informative message.
    _, pm = access_core_pm
    pm.set_vertex_property(12345, 67890, "foo", "bar")
    out = capsys.readouterr().out
    assert "Vertex (12345, 67890) not found." in out


# --- Tests for viewing properties -----------------------------------------

def test_view_node_properties_returns_expected(access_core_pm):
    # view_node_properties should return the raw property as well as decoded layer/node information.
    core, pm = access_core_pm
    layer_code = core._map_layer("L1")
    node_id_int = core._map_node_id("A")
    props = pm.view_node_properties(layer_code, node_id_int)
    assert props["p"] == 1
    assert props["decoded_layer"] == "L1"
    assert props["decoded_node_id"] == "A"
    # internal hashes should also be present as keys
    assert "layer_hash" in props and "node_id_hash" in props

def test_view_node_properties_unknown_vertex_returns_empty(capsys):
    # Asking to view a non-existent vertex should return empty dict and print an error.
    core = OnionNetGraph()
    pm = OnionNetPropertyManager(core)
    props = pm.view_node_properties(0, 0)
    assert props == {}
    out = capsys.readouterr().out
    assert "Vertex not found." in out

def test_view_node_properties_by_names_verbose(pm_and_graph_simple, capsys):
    # view_node_properties_by_names with verbose=True should print formatted output and return the same dict.
    core, pm = pm_and_graph_simple
    # decode grp so the decoded field exists
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    props = pm.view_node_properties_by_names("0", "A", verbose=True)
    out = capsys.readouterr().out
    assert "Properties for (0, A):" in out
    # ensure the returned dict contains expected entries
    assert 'grp' in props
    assert 'grp_decoded' in props
    assert 'decoded_layer' in props and props['decoded_layer'] == "0"
    assert 'decoded_node_id' in props and props['decoded_node_id'] == "A"


# --- Tests for label property creation ------------------------------------

def test_create_node_label_property_creates_and_formats(pm_and_graph_simple):
    # create_node_label_property should produce a string like "layer:node_id"
    core, pm = pm_and_graph_simple
    pm.create_node_label_property('node_label')
    assert 'node_label' in core.graph.vp
    v = pm.get_vertex_by_name_tuple("0", "A")
    label = core.graph.vp['node_label'][v]
    assert label == "0:A"  # exact format for this fixture

def test_create_node_label_property_idempotent(pm_and_graph_simple, capsys):
    # invoking create_node_label_property when it already exists should not error and should print a notice.
    core, pm = pm_and_graph_simple
    pm.create_node_label_property('node_label')
    # call again
    pm.create_node_label_property('node_label')
    out = capsys.readouterr().out
    assert "already exists" in out
    # label remains correct
    v = pm.get_vertex_by_name_tuple("0", "A")
    assert core.graph.vp['node_label'][v] == "0:A"



##### DECODING PROPERTIES

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


# --- Advanced / robustness tests ------------------------------------------

def test_roundtrip_decode_vertex_consistency(pm_and_graph_simple):
    # Round-trip decoding: decode then re-encode should recover original integer codes.
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    decoded_vals = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    inv_map = {v: k for k, v in core.vertex_categorical_mappings['grp']['int_to_str'].items()}
    reencoded = [inv_map.get(val) for val in decoded_vals]
    original_codes = [core.graph.vp['grp'][v] for v in core.graph.vertices()]
    assert reencoded == original_codes


def test_decode_numeric_with_custom_mapping(pm_and_graph_simple):
    # Decode a non-categorical numeric property by supplying an explicit mapping_dict.
    core, pm = pm_and_graph_simple
    # create a numeric vertex property manually
    vp = core.graph.new_vertex_property("int")
    for i, v in enumerate(core.graph.vertices()):
        vp[v] = i  # distinct codes 0,1,2
    core.graph.vp["num_code"] = vp
    custom_map = {0: "zero", 1: "one", 2: "two"}
    pm.decode_property_labels(
        encoded_prop_type='v',
        encoded_prop_name='num_code',
        new_prop_name='num_code_decoded',
        mapping_dict=custom_map,
        default_label='unknown'
    )
    decoded = [core.graph.vp['num_code_decoded'][v] for v in core.graph.vertices()]
    assert set(decoded) == {"zero", "one", "two"}


def test_bulk_decode_missing_encoded_prop_raises(pm_and_graph_simple):
    # DataFrame has an object column not present in categorical mappings → expect failure.
    core, pm = pm_and_graph_simple
    df = pd.DataFrame({"unknown": ["a", "b", "c"]})
    with pytest.raises(KeyError):
        pm.decode_property_labels_bulk(df, encoded_prop_type='v')


def test_bulk_decode_cleaned_name_collision(pm_and_graph_simple):
    # Two original column names collapse to the same cleaned decoded name; ensure at least one decoded prop appears.
    core, pm = pm_and_graph_simple
    # Prepare conflicting encoded props: "A B" and "a_b"
    core.vertex_categorical_mappings["A B"] = {
        "str_to_int": {"x": 0}, "int_to_str": {0: "x"}
    }
    vp1 = core.graph.new_vertex_property("int")
    for v in core.graph.vertices():
        vp1[v] = 0
    core.graph.vp["A B"] = vp1

    core.vertex_categorical_mappings["a_b"] = {
        "str_to_int": {"y": 0}, "int_to_str": {0: "y"}
    }
    vp2 = core.graph.new_vertex_property("int")
    for v in core.graph.vertices():
        vp2[v] = 0
    core.graph.vp["a_b"] = vp2

    df = pd.DataFrame({
        "A B": ["x", "x", "x"],
        "a_b": ["y", "y", "y"]
    })
    pm.decode_property_labels_bulk(df, encoded_prop_type='v')
    # Cleaned name is "a_b_decoded"
    assert "a_b_decoded" in core.graph.vp


def test_decode_idempotent(pm_and_graph_simple):
    # Running decode twice with same target property should produce identical results (overwrite case).
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    first = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    # mutate first to ensure overwrite happens
    core.graph.vp['grp_decoded'][core.graph.vertex(0)] = "OVERRIDE"
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    second = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    assert "OVERRIDE" not in second
    assert first == second


def test_partial_custom_mapping_with_default(pm_and_graph_simple):
    # Provide mapping that covers only some codes; unmatched codes use default_label.
    core, pm = pm_and_graph_simple
    grp_map = core.vertex_categorical_mappings['grp']['str_to_int']
    x_code = grp_map['x']
    custom_map = {x_code: "EX"}  # no mapping for 'y'
    pm.decode_property_labels('v', 'grp', mapping_dict=custom_map, default_label='MISSING', new_prop_name='grp_partial')
    decoded = [core.graph.vp['grp_partial'][v] for v in core.graph.vertices()]
    assert "EX" in decoded
    assert "MISSING" in decoded


def test_default_label_non_string(pm_and_graph_simple):
    core, pm = pm_and_graph_simple
    # Remove 'y' from the mapping so its code falls back to default_label
    orig_int_to_str = core.vertex_categorical_mappings["grp"]["int_to_str"]
    # Suppose 'y' has some code; drop it so we have a missing code
    y_code = core.vertex_categorical_mappings["grp"]["str_to_int"]["y"]
    core.vertex_categorical_mappings["grp"]["int_to_str"] = {k: v for k, v in orig_int_to_str.items() if k != y_code}

    pm.decode_property_labels('v', 'grp', default_label=123, new_prop_name='grp_default_int')
    decoded = [core.graph.vp['grp_default_int'][v] for v in core.graph.vertices()]
    # Expect at least one default label "123" (stringified)
    assert "123" in decoded


def test_bulk_decode_wrong_dimension(pm_and_graph_simple):
    # Passing an edge-like column to vertex decode bulk should error (dimension mismatch).
    core, pm = pm_and_graph_simple
    # 'lbl' is an edge property, attempt to decode it as vertex
    df = pd.DataFrame({"lbl": ["foo", "bar"]})
    with pytest.raises(ValueError):
        pm.decode_property_labels_bulk(df, encoded_prop_type='v')


def test_decode_after_mapping_mutation(pm_and_graph_simple):
    # Mutating the underlying categorical mapping between decodes should reflect in new decoded prop.
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    # change mapping for 'x' to something else
    x_code = core.vertex_categorical_mappings['grp']['str_to_int']['x']
    core.vertex_categorical_mappings['grp']['int_to_str'][x_code] = "Z"
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded2')
    decoded1 = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    decoded2 = [core.graph.vp['grp_decoded2'][v] for v in core.graph.vertices()]
    assert decoded1 != decoded2


def test_integration_view_node_properties_includes_decoded(pm_and_graph_simple):
    # After decoding, view_node_properties_by_names should still work and include decoded layer/node info.
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    props = pm.view_node_properties_by_names("0", "A", verbose=False)
    assert 'grp' in props
    assert 'decoded_layer' in props
    assert 'decoded_node_id' in props


@pytest.mark.slow
def test_high_cardinality_vertex_decode():
    # Stress test: create high-cardinality vertex categorical prop and decode it.
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)
    N = 3000  # moderately large to keep test reasonable
    cats = [f"cat_{i}" for i in range(N)]
    df = pd.DataFrame({
        "node_id": [str(i) for i in range(N)],
        "layer": ["0"] * N,
        "huge": cats
    })
    bldr.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=["huge"], drop_na=False)
    pm = OnionNetPropertyManager(core)
    pm.decode_property_labels('v', 'huge', new_prop_name='huge_decoded')
    decoded = {core.graph.vp['huge_decoded'][v] for v in core.graph.vertices() if isinstance(core.graph.vp['huge_decoded'][v], str)}
    # Expect at least all unique strings to be present in decoded
    assert len({d for d in decoded if d.startswith("cat_")}) >= N


# @pytest.mark.slow
# def test_high_cardinality_vertex_decode_huge():
#     # Stress test: create high-cardinality vertex categorical prop and decode it.
#     core = OnionNetGraph()
#     bldr = OnionNetBuilder(core)
#     N = 10_000_000  # large test
#     cats = [f"cat_{i}" for i in range(N)]
#     df = pd.DataFrame({
#         "node_id": [str(i) for i in range(N)],
#         "layer": ["0"] * N,
#         "huge": cats
#     })
#     bldr.add_vertices_from_dataframe(df, "node_id", "layer", property_cols=["huge"], drop_na=False)
#     pm = OnionNetPropertyManager(core)
#     pm.decode_property_labels('v', 'huge', new_prop_name='huge_decoded')
#     decoded = {core.graph.vp['huge_decoded'][v] for v in core.graph.vertices() if isinstance(core.graph.vp['huge_decoded'][v], str)}
#     # Expect at least all unique strings to be present in decoded
#     assert len({d for d in decoded if d.startswith("cat_")}) >= N


def test_decode_overwrites_existing(pm_and_graph_simple):
    # Decoding into an existing decoded property should restore correct values, not preserve manual overrides.
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    # manually override one decoded entry
    first_v = list(core.graph.vertices())[0]
    core.graph.vp['grp_decoded'][first_v] = "OVERRIDE"
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    decoded = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    assert "OVERRIDE" not in decoded


# --- Fixtures reused or slightly extended for edge tests -------------------

@pytest.fixture
def pm_and_graph_edge_extra():
    """
    Build a graph with one categorical edge property 'lbl' and one numeric edge property 'w'.
    This lets us test decoding and skipping of numeric vs object edge props.
    """
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)
    # seed vertices so edges have valid endpoints
    df_nodes = pd.DataFrame({
        "node_id": ["A", "B", "C"],
        "layer":   ["0", "0", "0"]
    })
    bldr.add_vertices_from_dataframe(df_nodes, "node_id", "layer", property_cols=None, drop_na=False)

    # seed edges with a categorical 'lbl' and numeric 'w'
    df_edges = pd.DataFrame({
        "source_id":    ["A", "B", "C"],
        "source_layer": ["0", "0", "0"],
        "target_id":    ["B", "C", "A"],
        "target_layer": ["0", "0", "0"],
        "lbl":          ["foo", "bar", "baz"],
        "w":            [1.0, 2.0, 3.0]
    })
    bldr.add_edges_from_dataframe(df_edges,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["lbl", "w"], drop_na=False)

    pm = OnionNetPropertyManager(core)
    return core, pm


# --- Additional edge-specific tests ---------------------------------------

def test_decode_edge_property_labels_default(pm_and_graph_simple):
    # Ensure that decoding an existing categorical edge property without overrides produces *_decoded
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('e', 'lbl')  # default decoding
    assert 'lbl_decoded' in core.graph.ep
    decoded = [core.graph.ep['lbl_decoded'][e] for e in core.graph.edges()]
    # original labels should appear (foo, bar)
    assert set(decoded) >= {"foo", "bar"}


def test_decode_edge_missing_codes_default_label(pm_and_graph_simple):
    # If the mapping lacks some codes, they should fall back to default_label
    core, pm = pm_and_graph_simple
    # remove 'bar' from mapping so its code maps to default
    int_to_str = core.edge_categorical_mappings["lbl"]["int_to_str"]
    # find code for 'bar' via reverse lookup
    str_to_int = core.edge_categorical_mappings["lbl"]["str_to_int"]
    bar_code = str_to_int["bar"]
    core.edge_categorical_mappings["lbl"]["int_to_str"] = {k: v for k, v in int_to_str.items() if k != bar_code}
    pm.decode_property_labels('e', 'lbl', new_prop_name="lbl2", default_label="UNKNOWN")
    decoded = [core.graph.ep["lbl2"][e] for e in core.graph.edges()]
    # Expect at least one "UNKNOWN" because 'bar' was dropped
    assert "UNKNOWN" in decoded


def test_decode_edge_wrong_dimension_error(pm_and_graph_simple):
    # Trying to decode an edge property as a vertex property should raise a clear error
    core, pm = pm_and_graph_simple
    df = pd.DataFrame({"lbl": ["foo", "bar", "baz"]})
    # Because 'lbl' exists only as an edge prop, decoding it as vertex should either KeyError or ValueError;
    # we expect the implementation to catch dimension mismatch and raise ValueError.
    with pytest.raises((ValueError, KeyError)):
        pm.decode_property_labels('v', 'lbl')


def test_bulk_decode_edge_props_and_skip_vertex(pm_and_graph_edge_extra, capsys):
    # Bulk decode on edges: object-type 'lbl' should be decoded, numeric 'w' skipped
    core, pm = pm_and_graph_edge_extra
    df = pd.DataFrame({
        "lbl": ["foo", "bar", "baz"],
        "w":   [1.0, 2.0, 3.0]
    })
    pm.decode_property_labels_bulk(df, encoded_prop_type='e')
    out = capsys.readouterr().out
    assert "w prop left as is" in out  # numeric skip message
    assert "lbl_decoded" in core.graph.ep
    vals = [core.graph.ep["lbl_decoded"][e] for e in core.graph.edges()]
    assert set(vals) == {"foo", "bar", "baz"}


def test_bulk_decode_missing_encoded_edge_prop_raises(pm_and_graph_edge_extra):
    # If the DataFrame refers to an edge property that doesn't exist on the graph,
    # bulk decoding should surface an error for that missing encoded property.
    core, pm = pm_and_graph_edge_extra
    df = pd.DataFrame({"unknown": ["a", "b", "c"]})
    with pytest.raises((KeyError, ValueError)):
        pm.decode_property_labels_bulk(df, encoded_prop_type='e')


def test_decode_edge_override_mapping_and_defaults(pm_and_graph_simple):
    # Provide a custom mapping that only maps one code; others should use default
    core, pm = pm_and_graph_simple
    # Determine original codes
    str_to_int = core.edge_categorical_mappings["lbl"]["str_to_int"]
    foo_code = str_to_int["foo"]
    custom_map = {foo_code: "F"}  # only 'foo' mapped
    pm.decode_property_labels(
        encoded_prop_type='e',
        encoded_prop_name='lbl',
        new_prop_name='lbl_custom',
        mapping_dict=custom_map,
        default_label="OTHER"
    )
    decoded = [core.graph.ep["lbl_custom"][e] for e in core.graph.edges()]
    # Must contain "F" for foo and "OTHER" for bar
    assert "F" in decoded and "OTHER" in decoded


def test_edge_property_type_conflict_reverse(builder_and_core, toy_nodes, toy_edges):
    # Ingest strength as categorical first, then attempt to add it again as numeric: should error
    bldr, core = builder_and_core
    # seed vertices
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    # first ingest strength as categorical (string_override True)
    bldr.add_edges_from_dataframe(
        toy_edges,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["strength"], drop_na=False, string_override=True
    )
    # now attempt to ingest same property again as numeric explicitly
    with pytest.raises(ValueError):
        bldr.add_edges_from_dataframe(
            toy_edges,
            source_id_col="source_id", source_layer_col="source_layer",
            target_id_col="target_id", target_layer_col="target_layer",
            property_cols=["strength"], drop_na=False, property_types={"strength": "int"}
        )


def test_edge_extreme_numeric_properties_still_show(builder_and_core, toy_nodes):
    # Edge numeric properties with extreme values (inf, -inf, nan) should propagate
    bldr, core = builder_and_core
    # seed nodes
    bldr.add_vertices_from_dataframe(toy_nodes, "node_id", "layer", drop_na=False)
    df = pd.DataFrame({
        "source_id":    ["A", "A", "B"],
        "source_layer": ["0", "0", "0"],
        "target_id":    ["B", "C", "C"],
        "target_layer": ["0", "1", "1"],
        "w":            [np.inf, -np.inf, np.nan]
    })
    bldr.add_edges_from_dataframe(df,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["w"], drop_na=False
    )
    ep = core.graph.ep["w"].a
    # ensure extreme values are preserved in order
    assert np.isposinf(ep[0])
    assert np.isneginf(ep[1])
    assert np.isnan(ep[2])


#### Tests of possible Vertex-Edge mixup causes ####
"""
    Summary of failure modes to be guarded against here:
    •	Normalization mismatches leading to unintended splits/merges of vertices.
    •	Index misalignment when filtering invalid edges, causing wrong properties to stick to surviving edges.
    •	Overwriting mappings on duplicate inserts that hide previously referenced vertices (edges still point to the old one).
    •	Name collision confusion between vertex and edge properties sharing the same key.
    •	Layer encoding errors that enable invalid cross-layer edges silently.
    •	Directionality collapse if source/target are confused.
    •	Scale-related statistical drift where rare malformed rows produce unexpected graph structure.
"""


def test_whitespace_id_splits_or_collapses(builder_and_core):
    bldr, core = builder_and_core
    # Three nodes that differ only by whitespace
    df = pd.DataFrame({
        "node_id": ["A", " A", "A "],
        "layer": ["0", "0", "0"]
    })
    bldr.add_vertices_from_dataframe(df, "node_id", "layer", drop_na=False)
    # If your system is meant to DE-duplicate by trimming, expect 1; otherwise 3
    keys = set(core.custom_id_to_vertex_index.keys())
    assert len(keys) in (1, 3)  # adapt to desired policy
    # Create edges referencing different forms and make sure they attach to the expected vertex identities
    edges = pd.DataFrame({
        "source_id": ["A", " A", "A "],
        "source_layer": ["0"]*3,
        "target_id": ["A", "A", " A"],
        "target_layer": ["0"]*3,
        "strength": [1,2,3]
    })
    bldr.add_edges_from_dataframe(edges, "source_id","source_layer", "target_id","target_layer",
                                 property_cols=["strength"], drop_na=False)
    # All edges should exist and not unintentionally collapse if not desired
    assert core.graph.num_edges() == 3


def test_edge_property_alignment_after_invalid_filtering(builder_and_core):
    # Edge with one invalid endpoint should be dropped, and properties for the kept ones remain correct.
    bldr, core = builder_and_core
    # seed only A and B
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]}),
        "node_id","layer", drop_na=False
    )
    df_e = pd.DataFrame({
        "source_id":    ["A","C","A"],  # 'C' is missing
        "source_layer": ["0","0","0"],
        "target_id":    ["B","B","B"],
        "target_layer": ["0","0","0"],
        "strength":     [100,200,300]
    })
    bldr.add_edges_from_dataframe(df_e, "source_id","source_layer", "target_id","target_layer",
                                  property_cols=["strength"], drop_na=False, 
                                  consider_props_in_duplicate=True)
    # Only A->B edges (strength 100 and 300) should survive
    assert core.graph.num_edges() == 2
    strengths = sorted(core.graph.ep["strength"].a.tolist())
    assert strengths == [100, 300]


def test_duplicate_vertex_overwrite_preserves_existing_edges(builder_and_core):
    bldr, core = builder_and_core
    # initial A->B
    nodes = pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]})
    edge = pd.DataFrame({
        "source_id":["A"],"source_layer":["0"],
        "target_id":["B"],"target_layer":["0"],
        "p": [1]
    })
    bldr.grow_onion(nodes, edge, node_prop_cols=[], edge_prop_cols=["p"],
                    drop_na=False, drop_duplicates=False)
    # record original source vertex index (A)
    original_idx = core.custom_id_to_vertex_index[(core._map_layer("0"), core._map_node_id("A"))]
    # add duplicate A with extra property
    dup_node = pd.DataFrame({"node_id":["A"],"layer":["0"], "newprop":[42]})
    bldr.add_vertices_from_dataframe(dup_node, "node_id","layer", property_cols=["newprop"], drop_na=False, drop_duplicates=False)
    # custom_id_to_vertex_index now maps to the new A
    new_idx = core.custom_id_to_vertex_index[(core._map_layer("0"), core._map_node_id("A"))]
    assert new_idx != original_idx
    # existing edge should still have source = original_idx
    e = list(core.graph.edges())[0]
    assert e.source() == original_idx


def test_vertex_and_edge_property_name_separation(builder_and_core):
    bldr, core = builder_and_core
    # seed vertex with 'weight'
    df_n = pd.DataFrame({"node_id":["A","B"],"layer":["0","0"], "weight":[1,2]})
    bldr.add_vertices_from_dataframe(df_n, "node_id","layer", property_cols=["weight"], drop_na=False)
    # seed edge with same prop name
    bldr.add_edges_from_dataframe(
        pd.DataFrame({
            "source_id":["A"],
            "source_layer":["0"],
            "target_id":["B"],
            "target_layer":["0"],
            "weight":[99]
        }),
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["weight"], drop_na=False
    )
    # vertex weight for A != edge weight
    v_idx = core.custom_id_to_vertex_index[(core._map_layer("0"), core._map_node_id("A"))]
    assert core.graph.vp["weight"][core.graph.vertex(v_idx)] == 1
    # edge weight is 99
    e = list(core.graph.edges())[0]
    assert core.graph.ep["weight"][e] == 99


def test_directional_edges_are_distinct(builder_and_core):
    bldr, core = builder_and_core
    # seed vertices
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]}),
        "node_id","layer", drop_na=False
    )
    # A->B with strength=1; B->A with strength=2
    edges = pd.DataFrame({
        "source_id":    ["A","B"],
        "source_layer": ["0","0"],
        "target_id":    ["B","A"],
        "target_layer": ["0","0"],
        "strength":     [1,2]
    })
    bldr.add_edges_from_dataframe(edges, "source_id","source_layer",
                                  "target_id","target_layer",
                                  property_cols=["strength"], drop_na=False)
    assert core.graph.num_edges() == 2
    strengths = sorted(core.graph.ep["strength"].a.tolist())
    assert strengths == [1,2]


def test_cross_layer_inconsistency_detection(builder_and_core):
    bldr, core = builder_and_core
    # nodes on layers 0 and 1
    df_n = pd.DataFrame({
        "node_id":["A","B"],
        "layer":["0","1"]
    })
    bldr.add_vertices_from_dataframe(df_n, "node_id","layer", drop_na=False)
    # Edge that incorrectly claims A is on layer "1" should be dropped if source/target mismatch
    bad_edge = pd.DataFrame({
        "source_id":["A"],
        "source_layer":["1"],  # mismatched layer
        "target_id":["B"],
        "target_layer":["1"],
    })
    bldr.add_edges_from_dataframe(bad_edge, "source_id","source_layer","target_id","target_layer",
                                  property_cols=None, drop_na=False)
    # Should have 0 edges because A@1 doesn't exist
    assert core.graph.num_edges() == 0
    # Also sanity check stored layer_hash matches original
    for v in core.graph.vertices():
        layer_code = core.graph.vp["layer_hash"][v]
        # decode back to name and ensure it's consistent
        name = core.layer_code_to_name.get(layer_code)
        assert name in {"0","1"}


def test_large_scale_near_collision_invariants(builder_and_core):
    bldr, core = builder_and_core
    # create near-duplicate ids
    base_ids = [f"node{i}" for i in range(100)]
    warped = [i if idx % 2 == 0 else f"{i} " for idx, i in enumerate(base_ids)]
    layers = ["0","1","2"]
    df_n = pd.DataFrame({
        "node_id": np.random.choice(warped, size=100),
        "layer":   np.random.choice(layers, size=100)
    })
    # add them
    bldr.add_vertices_from_dataframe(df_n, "node_id", "layer", drop_na=False)
    # random edges, some with invalid ids
    s_ids = np.random.choice(warped + ["nonexistent"], size=500)
    t_ids = np.random.choice(warped + ["missing"], size=500)
    s_layers = np.random.choice(layers, size=500)
    t_layers = np.random.choice(layers, size=500)
    df_e = pd.DataFrame({
        "source_id": s_ids,
        "source_layer": s_layers,
        "target_id": t_ids,
        "target_layer": t_layers
    })
    bldr.add_edges_from_dataframe(df_e, "source_id","source_layer","target_id","target_layer",
                                  property_cols=None, drop_na=False)
    # invariants
    for e in core.graph.edges():
        assert e.source() in core.vertex_index_to_custom_id
        assert e.target() in core.vertex_index_to_custom_id


def test_edges_before_vertices_do_not_create_them(builder_and_core):
    # Edges referencing missing vertices should not create vertices; only appear when endpoints exist.
    bldr, core = builder_and_core
    # attempt to add edge A->B before any vertices exist
    df_e = pd.DataFrame({
        "source_id": ["A"],
        "source_layer": ["0"],
        "target_id": ["B"],
        "target_layer": ["0"],
        "strength": [5]
    })
    bldr.add_edges_from_dataframe(df_e, "source_id", "source_layer", "target_id", "target_layer",
                                  property_cols=["strength"], drop_na=False)
    assert core.graph.num_edges() == 0  # nothing added
    # add only A
    bldr.add_vertices_from_dataframe(pd.DataFrame({"node_id":["A"],"layer":["0"]}), "node_id", "layer", drop_na=False)
    bldr.add_edges_from_dataframe(df_e, "source_id", "source_layer", "target_id", "target_layer",
                                  property_cols=["strength"], drop_na=False)
    assert core.graph.num_edges() == 0  # still cannot add, B missing
    # add B and now edge should appear
    bldr.add_vertices_from_dataframe(pd.DataFrame({"node_id":["B"],"layer":["0"]}), "node_id", "layer", drop_na=False)
    bldr.add_edges_from_dataframe(df_e, "source_id", "source_layer", "target_id", "target_layer",
                                  property_cols=["strength"], drop_na=False)
    assert core.graph.num_edges() == 1


def test_edge_deduplication_with_and_without_props_considered(builder_and_core):
    # If consider_props_in_duplicate=True, edges with same endpoints but different prop values survive separately;
    # otherwise they collapse.
    bldr, core = builder_and_core
    # seed endpoints A,B
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]}),
        "node_id","layer", drop_na=False
    )
    # create 10 A->B edges with property 'p' varying
    edge_rows = []
    for i in range(10):
        edge_rows.append({
            "source_id": "A", "source_layer": "0",
            "target_id": "B", "target_layer": "0",
            "p": i
        })
    df = pd.DataFrame(edge_rows)
    # with props considered: expect 10 edges
    bldr.add_edges_from_dataframe(df, "source_id", "source_layer",
                                  "target_id", "target_layer",
                                  property_cols=["p"], drop_na=False,
                                  consider_props_in_duplicate=True)
    assert core.graph.num_edges() == 10
    # cleanup and re-add without considering props: expect collapse to 1 edge (dedup)
    # reset graph
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]}),
        "node_id","layer", drop_na=False
    )
    bldr.add_edges_from_dataframe(df, "source_id", "source_layer",
                                  "target_id", "target_layer",
                                  property_cols=["p"], drop_na=False,
                                  consider_props_in_duplicate=False)
    assert core.graph.num_edges() == 1


def test_edge_roundtrip_categorical_mapping(pm_and_graph_simple):
    # Decode edge categorical prop and re-encode; should recover original integer codes.
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('e', 'lbl', new_prop_name='lbl_decoded')
    decoded_vals = [core.graph.ep['lbl_decoded'][e] for e in core.graph.edges()]
    # build reverse map (string -> code) from original str_to_int
    str_to_int = core.edge_categorical_mappings['lbl']['str_to_int']
    reencoded = [str_to_int[val] for val in decoded_vals]
    original_codes = [core.graph.ep['lbl'][e] for e in core.graph.edges()]
    assert reencoded == original_codes


def test_bulk_decode_idempotent(pm_and_graph_simple):
    # Running bulk decode twice should not change the decoded property after the first run.
    core, pm = pm_and_graph_simple
    df = pd.DataFrame({
        "grp": ["x", "y", "x"]
    })
    pm.decode_property_labels_bulk(df, encoded_prop_type='v')
    first = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    pm.decode_property_labels_bulk(df, encoded_prop_type='v')
    second = [core.graph.vp['grp_decoded'][v] for v in core.graph.vertices()]
    assert first == second


def test_whitespace_layer_current_behavior(builder_and_core):
    # Current behavior: layers with different whitespace are treated as distinct (no normalization).
    bldr, core = builder_and_core
    df = pd.DataFrame({
        "node_id": ["X", "Y", "Z"],
        "layer": ["1", " 1", "1 "]
    })
    bldr.add_vertices_from_dataframe(df, "node_id", "layer", drop_na=False)
    codes = {core._map_layer(l) for l in ["1", " 1", "1 "]}
    # expect distinct because no trimming is implemented yet
    assert len(codes) == 3


def test_summary_invariants_complex(builder_and_core):
    # Combined scenario: duplicates, NAs, invalid edges; summary metrics should satisfy arithmetic invariants.
    bldr, core = builder_and_core
    # Nodes: include one NA, one duplicate
    df_nodes = pd.DataFrame({
        "node_id": ["A", "A", None],
        "layer": ["0", "0", "1"]
    })
    # Edges: some invalid (missing endpoints), duplicates
    df_edges = pd.DataFrame({
        "source_id":    ["A", "A", "X"],
        "source_layer": ["0", "0", "0"],
        "target_id":    ["A", "A", "0"],  # "0" is missing node
        "target_layer": ["0", "0", "0"],
    })
    bldr.grow_onion(df_nodes, df_edges,
                   node_prop_cols=[], edge_prop_cols=[],
                   drop_na=True, drop_duplicates=True, verbose=False)
    summary = bldr.summary()
    # parse counts
    def parse_line(line):
        parts = {k: int(v) for k, v in
                 [p.split("=", 1) for p in line.replace(" → ", ",").split(", ")]}
        return parts
    node_line, edge_line = summary.splitlines()
    n = parse_line(node_line)
    e = parse_line(edge_line)
    # invariants: in - dropped_na - deduped == final
    assert n["in"] - n["dropped_na"] - n["deduped"] == n["final"]
    assert e["in"] - e["dropped_invalid"] - e["deduped"] == e["final"]


def test_vertex_property_type_conflict_reverse(builder_and_core):
    # Ingest a vertex property as categorical (string_override), then try to ingest it as numeric; should error.
    bldr, core = builder_and_core
    df = pd.DataFrame({
        "node_id": ["A"],
        "layer": ["0"],
        "foo": ["bar"]
    })
    # first ingest as categorical
    bldr.add_vertices_from_dataframe(df, "node_id", "layer",
                                     property_cols=["foo"], drop_na=False, string_override=True)
    # now attempt to ingest again as numeric explicitly
    with pytest.raises(ValueError):
        bldr.add_vertices_from_dataframe(df, "node_id", "layer",
                                         property_cols=["foo"], drop_na=False,
                                         property_types={"foo": "int"})


def test_decode_vertex_and_edge_same_name_separation(builder_and_core):
    # Ensure vertex/edge decode with same base prop name do not collide in namespace.
    bldr, core = builder_and_core
    # seed vertex and edge both with categorical "label"
    df_nodes = pd.DataFrame({
        "node_id": ["A", "B"],
        "layer": ["0", "0"],
        "label": ["u", "v"]
    })
    bldr.add_vertices_from_dataframe(df_nodes, "node_id", "layer",
                                     property_cols=["label"], drop_na=False)
    df_edges = pd.DataFrame({
        "source_id": ["A"],
        "source_layer": ["0"],
        "target_id": ["B"],
        "target_layer": ["0"],
        "label": ["w"]
    })
    bldr.add_edges_from_dataframe(
        df_edges,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["label"], drop_na=False
    )
    pm = OnionNetPropertyManager(core)
    pm.decode_property_labels('v', 'label', new_prop_name='label_decoded')
    pm.decode_property_labels('e', 'label', new_prop_name='label_decoded')
    assert 'label_decoded' in core.graph.vp
    assert 'label_decoded' in core.graph.ep
    # vertex and edge decoded values differ appropriately
    v_vals = [core.graph.vp['label_decoded'][v] for v in core.graph.vertices()]
    e_vals = [core.graph.ep['label_decoded'][e] for e in core.graph.edges()]
    assert any(isinstance(x, str) for x in v_vals)
    assert any(isinstance(x, str) for x in e_vals)


def test_high_cardinality_edge_decode(builder_and_core):
    # Stress test: many edges with unique categorical labels get decoded correctly.
    bldr, core = builder_and_core
    # seed two vertices
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]}),
        "node_id","layer", drop_na=False
    )
    N = 3_000 #tested up to 300_000 previously
    labels = [f"lbl_{i}" for i in range(N)]
    edge_rows = []
    for i, lbl in enumerate(labels):
        edge_rows.append({
            "source_id": "A", "source_layer": "0",
            "target_id": "B", "target_layer": "0",
            "huge_lbl": lbl
        })
    df = pd.DataFrame(edge_rows)
    bldr.add_edges_from_dataframe(
        df,
        source_id_col="source_id", source_layer_col="source_layer",
        target_id_col="target_id", target_layer_col="target_layer",
        property_cols=["huge_lbl"], drop_na=False,
        consider_props_in_duplicate=True
    )
    pm = OnionNetPropertyManager(core)
    pm.decode_property_labels('e', 'huge_lbl', new_prop_name='huge_lbl_decoded')
    decoded = {core.graph.ep['huge_lbl_decoded'][e] for e in core.graph.edges()}
    assert len({d for d in decoded if isinstance(d, str) and d.startswith("lbl_")}) >= N


def test_edge_decode_mapping_mutation(pm_and_graph_simple):
    # Mutate edge categorical mapping between decodes to ensure two decoded props differ.
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('e', 'lbl', new_prop_name='lbl_decoded')
    # change mapping for 'foo'
    foo_code = core.edge_categorical_mappings['lbl']['str_to_int']['foo']
    core.edge_categorical_mappings['lbl']['int_to_str'][foo_code] = "Z"
    pm.decode_property_labels('e', 'lbl', new_prop_name='lbl_decoded2')
    first = [core.graph.ep['lbl_decoded'][e] for e in core.graph.edges()]
    second = [core.graph.ep['lbl_decoded2'][e] for e in core.graph.edges()]
    assert first != second


def test_bulk_decode_invalid_encoded_prop_type_raises(pm_and_graph_simple):
    # Passing invalid encoded_prop_type to bulk decode should error early.
    core, pm = pm_and_graph_simple
    df = pd.DataFrame({"grp": ["x", "y", "x"]})
    with pytest.raises(ValueError):
        pm.decode_property_labels_bulk(df, encoded_prop_type='z')


def test_decode_overwrite_after_deletion(pm_and_graph_simple):
    # If a decoded property is deleted manually, re-running decode recreates it correctly.
    core, pm = pm_and_graph_simple
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    assert 'grp_decoded' in core.graph.vp
    # delete and verify it's gone
    del core.graph.vp['grp_decoded']
    assert 'grp_decoded' not in core.graph.vp
    # re-run decode
    pm.decode_property_labels('v', 'grp', new_prop_name='grp_decoded')
    assert 'grp_decoded' in core.graph.vp


def test_default_label_none_fallback(pm_and_graph_simple):
    # Use default_label=None and ensure it becomes a stringified "None" in decoded output.
    core, pm = pm_and_graph_simple
    # remove "y" from mapping to force default usage
    y_code = core.vertex_categorical_mappings["grp"]["str_to_int"]["y"]
    int_to_str = core.vertex_categorical_mappings["grp"]["int_to_str"]
    core.vertex_categorical_mappings["grp"]["int_to_str"] = {k: v for k, v in int_to_str.items() if k != y_code}
    pm.decode_property_labels('v', 'grp', default_label=None, new_prop_name='grp_default_none')
    decoded = [core.graph.vp['grp_default_none'][v] for v in core.graph.vertices()]
    # Expect at least one "None" string because of vectorize otypes=str
    assert "None" in decoded


def test_decode_with_extra_keys_in_mapping(pm_and_graph_simple):
    # Provide a mapping dict with unused (extra) keys; decoding should succeed and ignore extras.
    core, pm = pm_and_graph_simple
    # create custom map with extra dummy code
    base_map = core.vertex_categorical_mappings["grp"]["str_to_int"]
    inverse = {v: k for k, v in base_map.items()}
    # add extraneous code 999 mapping to "dummy"
    custom_map = {**inverse, 999: "dummy"}
    pm.decode_property_labels('v', 'grp', mapping_dict=custom_map, new_prop_name='grp_extra_keys')
    decoded = [core.graph.vp['grp_extra_keys'][v] for v in core.graph.vertices()]
    # Should contain proper labels and not fail due to extra key
    assert any(d in {"x", "y"} for d in decoded)


def test_edge_summary_and_property_alignment_after_invalid_filtering(builder_and_core):
    # After filtering invalid edges, verify properties align and duplicates controlled when consider_props_in_duplicate=True.
    bldr, core = builder_and_core
    # only seed A and B
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B"],"layer":["0","0"]}),
        "node_id","layer", drop_na=False
    )
    # include one invalid edge (C missing) and duplicate with different strength
    df_e = pd.DataFrame({
        "source_id":    ["A","C","A"],
        "source_layer": ["0","0","0"],
        "target_id":    ["B","B","B"],
        "target_layer": ["0","0","0"],
        "strength":     [100,200,300]
    })
    bldr.add_edges_from_dataframe(df_e, "source_id","source_layer", "target_id","target_layer",
                                  property_cols=["strength"], drop_na=False,
                                  consider_props_in_duplicate=True)
    # Expect two surviving A->B edges with strengths 100 and 300
    assert core.graph.num_edges() == 2
    strengths = sorted(core.graph.ep["strength"].a.tolist())
    assert strengths == [100, 300]


# --- Tests for get_category_code ------------------------------------------

def test_get_category_code_vertex_and_edge(pm_and_graph_simple):
    """
    get_category_code should return distinct integer codes for known categories
    on both vertex and edge properties.
    """
    core, pm = pm_and_graph_simple

    # vertex‐side: 'grp' has categories 'x' and 'y'
    code_x = pm.get_category_code('grp', 'x', dim='v')
    code_y = pm.get_category_code('grp', 'y', dim='v')
    assert isinstance(code_x, int)
    assert isinstance(code_y, int)
    assert code_x != code_y

    # edge‐side: 'lbl' has categories 'foo' and 'bar'
    e_code_foo = pm.get_category_code('lbl', 'foo', dim='e')
    e_code_bar = pm.get_category_code('lbl', 'bar', dim='e')
    assert isinstance(e_code_foo, int)
    assert isinstance(e_code_bar, int)
    assert e_code_foo != e_code_bar

@pytest.mark.parametrize("bad_dim", ["", "x", None])
def test_get_category_code_invalid_dim_raises(access_core_pm, bad_dim):
    """
    Passing an invalid dim should raise ValueError.
    """
    core, pm = access_core_pm
    with pytest.raises(ValueError):
        pm.get_category_code('p', '1', dim=bad_dim)

def test_get_category_code_unknown_prop_raises(access_core_pm):
    """
    Asking for a property that has no categorical mapping should raise KeyError.
    """
    core, pm = access_core_pm
    with pytest.raises(KeyError):
        pm.get_category_code('no_such_prop', 'anything', dim='v')

def test_get_category_code_unknown_label_raises(pm_and_graph_simple):
    """
    Asking for a label not seen in the mapping should raise KeyError.
    """
    core, pm = pm_and_graph_simple
    with pytest.raises(KeyError):
        pm.get_category_code('grp', 'not_a_label', dim='v')


def test_decode_edge_property_alignment_respects_edge_index():
    """
    Old bug: decoding wrote labels by zipping `for e in g.edges()` with prop.a,
    assuming iteration order == edge-index order. That’s not guaranteed.
    This test constructs edges in an order that *differs* from iteration order,
    so a buggy implementation will swap labels.
    """
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)

    # Vertices A (0), B (1), C (2)
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B","C"], "layer":["0","0","0"]}),
        "node_id","layer", drop_na=False
    )

    # Add edges in this INSERTION order: B->C ('bar'), then A->B ('foo')
    df_e = pd.DataFrame({
        "source_id":    ["B",   "A"],
        "source_layer": ["0",   "0"],
        "target_id":    ["C",   "B"],
        "target_layer": ["0",   "0"],
        "lbl":          ["bar", "foo"],
    })
    bldr.add_edges_from_dataframe(
        df_e, "source_id","source_layer","target_id","target_layer",
        property_cols=["lbl"], drop_na=False
    )

    pm = OnionNetPropertyManager(core)
    pm.decode_property_labels('e', 'lbl', new_prop_name='lbl_decoded')

    g = core.graph
    lbl_codes = g.ep['lbl'].a
    int_to_str = core.edge_categorical_mappings['lbl']['int_to_str']

    # Collect expected label per *edge* via true edge index (source of truth)
    for e in g.edges():
        eidx = int(g.edge_index[e])             # internal edge index
        expected = int_to_str[int(lbl_codes[eidx])]
        got = g.ep['lbl_decoded'][e]
        assert got == expected, f"edge {int(e.source())}->{int(e.target())}: {got} != {expected}"

    # Make sure iteration order actually differs from insertion order (so this test really guards the bug)
    order_seen = [int(g.edge_index[e]) for e in g.edges()]
    assert order_seen != sorted(order_seen), \
        "Iteration order unexpectedly matches edge-index order; the regression might slip by on this platform."
    

def test_decode_edge_source_target_layers_map_correctly():
    """
    Ensure source_layer_decoded and target_layer_decoded come from the correct
    integer-coded columns for each edge (no accidental swapping or misalignment).
    """
    core = OnionNetGraph()
    bldr = OnionNetBuilder(core)

    # Layers and vertices
    bldr.add_vertices_from_dataframe(
        pd.DataFrame({"node_id":["A","B","C"], "layer":["L1","L2","L1"]}),
        "node_id","layer", drop_na=False
    )

    # Two cross-layer edges so src/tgt layers differ per edge
    edges = pd.DataFrame({
        "source_id":    ["A", "B"],
        "source_layer": ["L1","L2"],
        "target_id":    ["B", "C"],
        "target_layer": ["L2","L1"],
    })
    # Keep layer columns as edge properties so we can decode them
    bldr.add_edges_from_dataframe(
        edges, "source_id","source_layer","target_id","target_layer",
        property_cols=["source_layer","target_layer"], drop_na=False
    )

    pm = OnionNetPropertyManager(core)
    pm.decode_property_labels('e', 'source_layer', new_prop_name='source_layer_decoded')
    pm.decode_property_labels('e', 'target_layer', new_prop_name='target_layer_decoded')

    g = core.graph
    # Build expected strings directly from the integer edge props using the global layer map
    layer_map = core.layer_code_to_name
    src_codes = g.ep['source_layer'].a
    tgt_codes = g.ep['target_layer'].a

    for e in g.edges():
        eidx = int(g.edge_index[e])
        expected_src = layer_map[int(src_codes[eidx])]
        expected_tgt = layer_map[int(tgt_codes[eidx])]
        got_src = g.ep['source_layer_decoded'][e]
        got_tgt = g.ep['target_layer_decoded'][e]
        assert got_src == expected_src
        assert got_tgt == expected_tgt
        # if the codes differ for this edge, strings must differ too
        if src_codes[eidx] != tgt_codes[eidx]:
            assert got_src != got_tgt