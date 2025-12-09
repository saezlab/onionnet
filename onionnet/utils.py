"""Utility helpers for OnionNet (type inference, category mapping)."""

from __future__ import annotations

import numpy as np
import pandas as pd

#########################################
# Utility Functions
#########################################


def infer_property_type(value):
    """
    Infer a simple property type from a pandas Series or a single value.

    For pandas Series, the Series ``dtype`` is mapped to one of the
    following strings after inspection:

    - ``"int"`` for integer types
    - ``"float"`` for floating-point types
    - ``"bool"`` for boolean types
    - ``"string"`` for object or other types

    For individual values, ``isinstance`` checks are used to return the
    corresponding string.

    Parameters
    ----------
    value : pandas.Series | int | float | bool | str
        A pandas Series or a single sample value.

    Returns
    -------
    str
        One of: ``"int"``, ``"float"``, ``"bool"``, ``"string"``, ``"object"``.
    """
    # If value is a pandas Series, use its dtype
    if hasattr(value, "dtype"):
        if pd.api.types.is_integer_dtype(value.dtype):
            return "int"
        if pd.api.types.is_float_dtype(value.dtype):
            return "float"
        if pd.api.types.is_bool_dtype(value.dtype):
            return "bool"
        # For object or other dtypes, assume string
        return "string"

    # Fallback for single sample values
    if isinstance(value, int | np.integer):
        return "int"
    if isinstance(value, float | np.floating):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool | np.bool_):
        return "bool"
    return "object"


def map_categorical_property(prop_name, values, mapping: dict[str, int] | None = None):
    """
    Map categorical property values to unique integer codes.

    This function converts an iterable of categorical values into a NumPy array of integer codes. Each unique value
    is assigned a unique integer. If an initial mapping is provided, it will be used as the starting point; otherwise,
    a new mapping is created. The function returns both the array of integer codes and the mapping dictionary.

    Parameters
    ----------
    prop_name : str
        The name of the property being mapped (used for reference or debugging).
    values : iterable
        An array-like collection of categorical values to map.
    mapping : dict of (str, int) or None, optional
        Existing dictionary mapping categorical values to integer codes. Defaults to None.

    Returns
    -------
        tuple
            A tuple containing:
                - mapped_values (numpy.ndarray): A NumPy array of integer codes corresponding to each value in 'values'.
                - mapping dict of (str, int): The updated dictionary mapping each unique categorical value to its integer code.
    """
    # mark prop_name as intentionally unused in logic
    _ = prop_name
    if mapping is None:
        mapping = {}
    mapped_values = np.empty(len(values), dtype=np.int32)
    current_code = len(mapping)
    for i, val in enumerate(values):
        if val in mapping:
            mapped_values[i] = mapping[val]
        else:
            mapping[val] = current_code
            mapped_values[i] = current_code
            current_code += 1
    return mapped_values, mapping
