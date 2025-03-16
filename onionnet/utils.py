import numpy as np
import pandas as pd
from typing import List, Any, Dict

#########################################
# Utility Functions
#########################################

def infer_property_type(value):
    """
    Infer the property type based on a pandas Series or a single sample value.
    
    This function inspects the input value to determine its type. If the value is a pandas Series, it uses the Series
    dtype to infer the type and returns:
      - 'int' for integer types,
      - 'float' for floating-point types,
      - 'bool' for boolean types,
      - 'string' for object or other types.
    
    For individual sample values, the function checks the type using isinstance and returns the corresponding string.
    
    Parameters:
        value: A pandas Series or a single sample value (int, float, bool, or str).
    
    Returns:
        str: A string representing the inferred property type. Possible return values include 'int', 'float', 'bool', 'string', or 'object'.
    """
    # If value is a pandas Series, use its dtype
    if hasattr(value, 'dtype'):
        if pd.api.types.is_integer_dtype(value.dtype):
            return 'int'
        elif pd.api.types.is_float_dtype(value.dtype):
            return 'float'
        elif pd.api.types.is_bool_dtype(value.dtype):
            return 'bool'
        else:
            # For object or other dtypes, assume string
            return 'string'
    
    # Fallback for single sample values
    if isinstance(value, (int, np.integer)):
        return 'int'
    elif isinstance(value, (float, np.floating)):
        return 'float'
    elif isinstance(value, str):
        return 'string'
    elif isinstance(value, (bool, np.bool_)):
        return 'bool'
    else:
        return 'object'


def map_categorical_property(prop_name, values, mapping: Dict[str, int] = None):
    """
    Map categorical property values to unique integer codes.
    
    This function converts an iterable of categorical values into a NumPy array of integer codes. Each unique value
    is assigned a unique integer. If an initial mapping is provided, it will be used as the starting point; otherwise,
    a new mapping is created. The function returns both the array of integer codes and the mapping dictionary.
    
    Parameters:
        prop_name (str): The name of the property being mapped (used for reference or debugging).
        values (iterable): An array-like collection of categorical values to map.
        mapping (Dict[str, int], optional): An existing dictionary mapping categorical values to integer codes. Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - mapped_values (np.ndarray): A NumPy array of integer codes corresponding to each value in 'values'.
            - mapping (Dict[str, int]): The updated dictionary mapping each unique categorical value to its integer code.
    """
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