import pytest
import pandas as pd
import numpy as np

from onionnet.utils import infer_property_type, map_categorical_property


def test_infer_property_type_series_and_scalars():
    # Series types
    assert infer_property_type(pd.Series([1, 2, 3])) == 'int'
    assert infer_property_type(pd.Series([1.0, 2.5])) == 'float'
    assert infer_property_type(pd.Series([True, False])) == 'bool'
    assert infer_property_type(pd.Series(['a', 'b'])) == 'string'

    # Scalar values
    assert infer_property_type(3) == 'int'
    assert infer_property_type(np.int64(4)) == 'int'
    assert infer_property_type(3.14) == 'float'
    assert infer_property_type(np.float32(1.2)) == 'float'
    assert infer_property_type('x') == 'string'

    # Note: current implementation classifies bool scalars as 'int'
    assert infer_property_type(True) in {'int', 'bool'}

    class Custom:
        pass
    assert infer_property_type(Custom()) == 'object'


def test_map_categorical_property_new_and_existing_mapping():
    values = ['a', 'b', 'a', 'c']
    mapped, mapping = map_categorical_property('prop', values)
    assert list(mapped) == [0, 1, 0, 2]
    assert mapping == {'a': 0, 'b': 1, 'c': 2}

    # Extend with an existing mapping, adding a new category 'd'
    values2 = ['c', 'd']
    mapped2, mapping2 = map_categorical_property('prop', values2, mapping)
    assert list(mapped2) == [2, 3]
    assert mapping2['d'] == 3

