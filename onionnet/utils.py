import numpy as np
import pandas as pd
from typing import List, Any, Dict

#########################################
# Utility Functions
#########################################
def infer_property_type(value):
    """Infer property type from a pandas Series or a sample value."""
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
    """Map categorical values to integer codes."""
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