import graph_tool.all as gt
import graph_tool
import matplotlib.pyplot as plt
from typing import Dict, List, Any

import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
import numpy as np

import matplotlib.cm as cm  # To use color maps
from matplotlib.patches import Patch

# For layout compute or load function
import os
import pandas as pd
from graph_tool.all import sfdp_layout

"""
This module provides visualization utilities for the OnionNet project. It includes functions for generating graph layouts, 
assigning colors and shapes to nodes and edges based on their properties, drawing weight propagation graphs, and creating legends. 
These tools enable effective visual analysis and presentation of complex network data within an OnionNetGraph.
"""

#########################################
# Visualisation: Graph Layout and Styling
#########################################


def flatten_properties(nested_properties: List[Any]) -> List[str]:
    """
    Utility function to flatten a list of nested properties into a single list.
    Assumes that properties may be nested within sublists.
    
    Parameters:
    ----------
    nested_properties : list
        The list of potentially nested properties to be flattened.
    
    Returns:
    -------
    List[str]
        A flattened list of properties, with duplicates removed.
    """
    flat_list = []
    for item in nested_properties:
        if isinstance(item, list):
            flat_list.extend(flatten_properties(item))
        else:
            flat_list.append(str(item))  # Convert everything to string
    return flat_list


def create_node_labels(g: gt.Graph, property_map: gt.PropertyMap) -> gt.PropertyMap:
    """
    Creates a vertex property for node labels from potentially nested properties.
    
    Parameters:
    ----------
    g : graph_tool.Graph
        The graph containing the nodes.
    property_map : graph_tool.PropertyMap
        A property map containing the nested properties for each node.
    
    Returns:
    -------
    vertex_labels : graph_tool.PropertyMap
        A string property map with flattened and unique properties as node labels.
    """
    vertex_labels = g.new_vertex_property("string")

    for v in g.vertices():
        # Get the properties for the node
        node_properties = property_map[v]
        if node_properties:
            # Flatten the properties list
            flat_properties = flatten_properties(node_properties)
            # Convert to a set to remove duplicates (optional)
            unique_properties = set(flat_properties)
            # Join the properties into a string
            vertex_labels[v] = ", ".join(unique_properties)
        else:
            vertex_labels[v] = ""

    return vertex_labels


def color_nodes(
    g,
    prop_name,
    method="categorical",
    generate_legend=False,
    custom_colormap=None,
    custom_color_dict=None,
    zero_centred=False
):
    """
    Assign colors to nodes in a graph.

    Parameters:
    -------
    - g (Graph): The graph object where nodes are styled.
    - prop_name (str): The name of the vertex property used to determine colors.
    - method (str): 
        'categorical': assigns distinct colors for each unique category in the property.
        'continuous' : uses a color scale.
        'boolean'    : uses red if the property is True, grey if False.
    - generate_legend (bool): If True, generates a legend dictionary mapping categories to colors.
    - custom_colormap (Colormap or None): A custom matplotlib colormap for continuous values.
    - custom_color_dict (dict or None): A user-defined dictionary mapping property values to colors.
    - zero_centered (bool): If True (and method is 'continuous'), adjusts the normalization range so that
          zero is centered (i.e. using symmetric bounds [-abs(max_val), abs(max_val)]). Defaults to False. 

    Returns:
    -------
    - result (dict): A dictionary containing:
        - 'v_color' (PropertyMap): A vertex property map with RGBA color values.
        - 'legend' (dict): A dictionary mapping categories to colors (if generate_legend=True).
    """
    v_color = g.new_vertex_property("vector<double>")
    legend = {} if generate_legend else None

    # Handle custom color dictionary
    if custom_color_dict:
        for v in g.vertices():
            value = g.vp[prop_name][v]
            if value in custom_color_dict:
                v_color[v] = custom_color_dict[value]
            else:
                raise ValueError(f"Value '{value}' not found in custom_color_dict.")
        if generate_legend:
            legend = custom_color_dict

    # Handle colors with custom colormap or default colormap
    elif method == "categorical":
        categories = list(set(g.vp[prop_name]))
        colormap = custom_colormap or cm.tab10
        colormap_len = len(colormap.colors)
        color_map = {cat: colormap(i % colormap_len) for i, cat in enumerate(categories)}
        for v in g.vertices():
            category = g.vp[prop_name][v]
            v_color[v] = color_map[category][:3] + (1.0,)
        if generate_legend:
            legend = {cat: color_map[cat][:3] + (1.0,) for cat in categories}

    elif method == "continuous":
        values = [float(g.vp[prop_name][v]) for v in g.vertices()]
        min_val, max_val = min(values), max(values)
        # If the user wants, they can set the middle of the bar based on the max absolute value
        if zero_centred == True:
            abs_max = max(abs(min_val), abs(max_val))
            min_val = -abs_max
            max_val = abs_max    
        colormap = custom_colormap or cm.viridis
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap)
        for v in g.vertices():
            value = float(g.vp[prop_name][v])
            v_color[v] = scalar_map.to_rgba(value)[:3] + (1.0,)
        if generate_legend:
            legend = {"min_col": scalar_map.to_rgba(min_val), "max_col": scalar_map.to_rgba(max_val),
                      "min_val": min_val, "max_val": max_val}

    elif method == "boolean":
        for v in g.vertices():
            value = g.vp[prop_name][v]
            v_color[v] = (1.0, 0.0, 0.0, 1.0) if bool(value) else (0.5, 0.5, 0.5, 1.0)
        if generate_legend:
            legend = {"True": (1.0, 0.0, 0.0, 1.0), "False": (0.5, 0.5, 0.5, 1.0)}

    else:
        raise ValueError("Unsupported color method. Choose from: categorical, continuous, boolean.")

    return {"v_color": v_color, "legend_node_color": legend}


def shape_nodes(
    g,
    prop_name,
    shape_method=None,
    generate_legend=False,
    custom_shape_dict=None,
):
    """
    Assign shapes to nodes in a graph.

    Parameters:
    -------
    - g (Graph): The graph object where nodes are styled.
    - prop_name (str): The name of the vertex property used to determine shapes.
    - shape_method (str or None): If specified, assigns vertex shapes based on a property or method.
    - generate_legend (bool): If True, generates a legend dictionary mapping categories to shapes.
    - custom_shape_dict (dict or None): A user-defined dictionary mapping property values to shapes.

    Returns:
    -------
    - result (dict): A dictionary containing:
        - 'v_shape' (PropertyMap): A vertex property map with shape values.
        - 'legend' (dict): A dictionary mapping categories to shapes (if generate_legend=True).
    """
    v_shape = g.new_vertex_property("string")
    legend = {} if generate_legend else None

    # Handle custom shape dictionary
    if custom_shape_dict:
        for v in g.vertices():
            value = g.vp[prop_name][v]
            if value in custom_shape_dict:
                v_shape[v] = custom_shape_dict[value]
            else:
                raise ValueError(f"Value '{value}' not found in custom_shape_dict.")
        if generate_legend:
            legend = custom_shape_dict

    # Handle shapes with default assignment
    elif shape_method == "categorical":
        categories = list(set(g.vp[prop_name]))
        shapes = ["circle", "triangle", "square", "pentagon", "hexagon"]
        shape_map = {cat: shapes[i % len(shapes)] for i, cat in enumerate(categories)}
        for v in g.vertices():
            category = g.vp[prop_name][v]
            v_shape[v] = shape_map.get(category, "circle")
        if generate_legend:
            legend = shape_map

    elif shape_method == "boolean":
        for v in g.vertices():
            value = g.vp[prop_name][v]
            v_shape[v] = "triangle" if bool(value) else "square"
        if generate_legend:
            legend = {"True": "triangle", "False": "square"}

    return {"v_shape": v_shape, "legend_node_shape": legend}


def add_halo_to_node(
    g,
    node,
    halo_color=(1.0, 1.0, 0.0, 0.5),  # Default yellow halo
    halo_size_factor=1.5,
):
    """
    Add a halo to a specific node while styling the graph.

    Parameters:
    -------
    - g (Graph): The graph object.
    - node (Vertex): The specific vertex requiring a halo.
    - halo_color (tuple): RGBA color for the halo.
    - halo_size_factor (float): Size of the halo relative to the node size.

    Returns:
    -------
    - result (dict): A dictionary containing:
        - 'v_halo' (PropertyMap): Halo property map (only for the specific node).
        - 'v_halo_color' (PropertyMap): Halo colour as a property map (only for the specific node with halo).
    """
    # Initialize halo property
    v_halo = g.new_vertex_property("bool")
    v_halo_color = g.new_vertex_property("vector<double>")
    
    for v in g.vertices():
        #print(v)
        if v == node:  # Add a halo to the specified node
            v_halo[v] = True 
            v_halo_color[v] = halo_color
        else:  # No halo for other nodes
            v_halo[v] = False #(0, 0, 0, 0)  # Transparent / no halo

    return {"v_halo": v_halo, "v_halo_color": v_halo_color}


def set_node_sizes_and_text_by_depth(g, root, max_size=20, min_size=5, max_text_size=15, min_text_size=8):
    """
    Set node sizes and text sizes based on their depth in the tree.
    
    Parameters:
    - g (Graph): The graph object.
    - root (Vertex): The root vertex from which to calculate depths.
    - max_size (int): Maximum size for inner nodes (closer to the root).
    - min_size (int): Minimum size for outer nodes (further from the root).
    - max_text_size (int): Maximum text size for inner nodes.
    - min_text_size (int): Minimum text size for outer nodes.
    
    Returns:
    - v_size (PropertyMap): A vertex property map with sizes based on depth.
    - v_text_size (PropertyMap): A vertex property map for text sizes based on depth.
    """
    # TODO - text_size seems currently buggy in cairo, might need to fix or go back to just node size

    # Create property maps for storing node sizes and text sizes
    v_size = g.new_vertex_property("double")
    v_text_size = g.new_vertex_property("double")
    
    # Calculate depths of each node from the root
    depths = graph_tool.topology.shortest_distance(g, source=root, directed=False, weights=None)
    max_depth = np.max(depths)
    
    for v in g.vertices():
        depth = depths[v]
        
        # Scale node size based on depth
        v_size[v] = max_size - ((max_size - min_size) * (depth / max_depth))
        
        # Scale text size based on depth
        v_text_size[v] = max_text_size - ((max_text_size - min_text_size) * (depth / max_depth))
    
    return v_size, v_text_size


def get_legend(source, prop=None, ordered_cats=None, verbose=False, mode=None, custom_cmap=None, title: str=None, save_filename: str=None):
    """
    Generates a legend for a graph coloring.
    
    Parameters:
    -----------
    source: Either a graph (with vertex or edge properties) or a legend dictionary mapping categories to colors.
    prop: If source is a graph, the property name to extract values from (vertex or edge property).
    ordered_cats: Optional list specifying the order of categories in the legend (for categorical legends).
    verbose: If True, prints debug information.
    mode: Optional, 'categorical' or 'continuous'. If None, the function will infer the mode from the property type.
    custom_cmap: A custom matplotlib colormap to use for continuous legends. Defaults to viridis if not provided.
    
    Behavior:
    -----------    
    - If source is a dictionary:
      * If it contains keys 'min_col' and 'max_col', it is treated as a continuous legend dictionary and a colorbar is displayed.\n
      * Otherwise, it is treated as a categorical mapping from categories to colors.
    - If source is a graph object, the function extracts the property values from source.vp[prop] or source.ep[prop].\n
      If the property values are numeric (or mode is set to 'continuous'), a continuous colorbar is displayed.\n
      Otherwise, a categorical legend is constructed using a default colormap (tab10).
    """
    import matplotlib.cm as cm
    from matplotlib.patches import Patch
    
    # Case 1: source is a dictionary
    if isinstance(source, dict):
        if 'min_col' in source and 'max_col' in source:
            # Continuous legend dictionary provided\n
            min_val = source.get('min_val')
            max_val = source.get('max_val')
            if min_val is None or max_val is None:
                raise ValueError("Continuous legend dictionary must contain 'min_val' and 'max_val'.")
            cmap = custom_cmap if custom_cmap is not None else cm.viridis
            norm = plt.Normalize(vmin=min_val, vmax=max_val)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.figure(figsize=(6, 1))
            cbar = plt.colorbar(sm, orientation='horizontal')
            cbar.set_label(prop.capitalize() if prop else "Value")
            plt.show()
            return
        else:
            # Categorical legend dictionary provided\n
            legend_dict = source
            mode = 'categorical'
    else:
        # Case 2: source is assumed to be a graph object\n
        if prop is None:
            raise ValueError("When source is a graph, 'prop' must be provided.")
        # Determine mode if not explicitly provided\n
        if mode is None:
            if hasattr(source, "vp") and prop in source.vp:
                sample = next(iter(source.vp[prop]))
            elif hasattr(source, "ep") and prop in source.ep:
                sample = next(iter(source.ep[prop]))
            else:
                raise ValueError("Provided graph does not have the specified property.")
            mode = 'continuous' if isinstance(sample, (int, float)) else 'categorical'
        if mode == 'continuous':
            # Extract numeric values from the property\n
            if hasattr(source, "vp") and prop in source.vp:
                values = [float(x) for x in source.vp[prop]]
            elif hasattr(source, "ep") and prop in source.ep:
                values = [float(x) for x in source.ep[prop]]
            else:
                raise ValueError("Provided graph does not have the specified property.")
            min_val, max_val = min(values), max(values)
            cmap = custom_cmap if custom_cmap is not None else cm.viridis
            norm = plt.Normalize(vmin=min_val, vmax=max_val)
            sm = cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.figure(figsize=(6, 1))
            cbar = plt.colorbar(sm, orientation='horizontal')
            cbar.set_label(prop.capitalize() if prop else "Value")
            plt.show()
            return
        elif mode == 'categorical':
            if hasattr(source, "vp") and prop in source.vp:
                categories = set(source.vp[prop])
            elif hasattr(source, "ep") and prop in source.ep:
                categories = set(source.ep[prop])
            else:
                raise ValueError("Provided graph does not have the specified property.")
            # Use default colormap (tab10) for distinct color assignment\n
            legend_dict = {cat: cm.tab10(i % 10) for i, cat in enumerate(categories)}
            if verbose:
                print("Default legend dictionary:", legend_dict)
        else:
            raise ValueError("Mode must be either 'continuous' or 'categorical'.")
    
    # Categorical legend: create legend elements using patches\n
    if ordered_cats is not None:
        legend_elements = [Patch(facecolor=legend_dict[cat][:3] if isinstance(legend_dict[cat], (tuple, list)) else legend_dict[cat], label=cat)
                           for cat in ordered_cats if cat in legend_dict]
    else:
        legend_elements = [Patch(facecolor=(color[:3] if isinstance(color, (tuple, list)) else color), label=category)
                           for category, color in legend_dict.items()]
    
    plt.figure(figsize=(5, 3))
    plot_title = title if title is not None else (prop.capitalize() if prop is not None else "Legend")
    plt.legend(handles=legend_elements, title=plot_title, loc="center", frameon=False)
    plt.axis("off")
    # Save the figure as an SVG file
    if save_filename != None:
        plt.savefig(f"{save_filename}.svg", format="svg")
    plt.show()

    
def color_edges(g, prop_name, method="categorical", generate_legend=False, custom_colormap=None, custom_color_dict=None, zero_centred=False):
    """
    Assign colors to edges in a graph.
    
    Parameters:
    -----------
    g (Graph): The graph object where edges are styled.
    prop_name (str): The name of the edge property used to determine colors.
    method (str): 
        'categorical': assigns distinct colors for each unique category in the property.
        'continuous': uses a color scale.
        'boolean': uses red if the property is True, grey if False.
    generate_legend (bool): If True, generates a legend dictionary mapping categories to colors.
    custom_colormap (Colormap or None): A custom matplotlib colormap for continuous values.
    custom_color_dict (dict or None): A user-defined dictionary mapping property values to colors.
    zero_centred (bool): If True (and method is 'continuous'), adjusts the normalization range so that
        zero is centered. Defaults to False.
    
    Returns:
    --------
    dict: A dictionary containing:
        - 'e_color' (PropertyMap): An edge property map with RGBA color values.
        - 'legend_edge_color' (dict or None): A dictionary mapping categories to colors if generate_legend is True.
    """
    e_color = g.new_edge_property("vector<double>")
    legend = {} if generate_legend else None

    # Handle custom color dictionary
    if custom_color_dict:
        for e in g.edges():
            value = g.ep[prop_name][e]
            if value in custom_color_dict:
                e_color[e] = custom_color_dict[value]
            else:
                raise ValueError(f"Value '{value}' not found in custom_color_dict.")
        if generate_legend:
            legend = custom_color_dict

    elif method == "categorical":
        categories = list(set(g.ep[prop_name]))
        colormap = custom_colormap or cm.tab10
        colormap_len = len(colormap.colors)
        color_map = {cat: colormap(i % colormap_len) for i, cat in enumerate(categories)}
        for e in g.edges():
            category = g.ep[prop_name][e]
            e_color[e] = color_map[category][:3] + (1.0,)
        if generate_legend:
            legend = {cat: color_map[cat][:3] + (1.0,) for cat in categories}

    elif method == "continuous":
        values = [float(g.ep[prop_name][e]) for e in g.edges()]
        min_val, max_val = min(values), max(values)
        if zero_centred:
            abs_max = max(abs(min_val), abs(max_val))
            min_val = -abs_max
            max_val = abs_max
        colormap = custom_colormap or cm.viridis
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=colormap)
        for e in g.edges():
            value = float(g.ep[prop_name][e])
            e_color[e] = scalar_map.to_rgba(value)[:3] + (1.0,)
        if generate_legend:
            legend = {"min_col": scalar_map.to_rgba(min_val), "max_col": scalar_map.to_rgba(max_val),
                      "min_val": min_val, "max_val": max_val}

    elif method == "boolean":
        for e in g.edges():
            value = g.ep[prop_name][e]
            e_color[e] = (1.0, 0.0, 0.0, 1.0) if bool(value) else (0.5, 0.5, 0.5, 1.0)
        if generate_legend:
            legend = {"True": (1.0, 0.0, 0.0, 1.0), "False": (0.5, 0.5, 0.5, 1.0)}

    else:
        raise ValueError("Unsupported color method. Choose from: categorical, continuous, boolean.")

    return {"e_color": e_color, "legend_edge_color": legend}


def layout_by_layer(g, layer_prop_name='layer_decoded', spacing=50, epsilon=1e-2):
    """
    Create a 2D layout that places nodes in vertical columns based on their layer.
    Vertices in each layer are spaced out by 'spacing' units. If a layer has only one
    vertex, a small random offset (epsilon) is added to avoid a zero spread.
    """
    if layer_prop_name not in g.vp:
        raise KeyError(f"Vertex property '{layer_prop_name}' not found.")

    pos = g.new_vertex_property("vector<double>")
    layer_dict = {}
    for v in g.vertices():
        layer_val = g.vp[layer_prop_name][v]
        layer_dict.setdefault(layer_val, []).append(v)

    # Assign each unique layer an x coordinate.
    unique_layers = sorted(layer_dict.keys())
    layer_to_x = {layer_val: i * spacing for i, layer_val in enumerate(unique_layers)}

    for layer_val, vertices in layer_dict.items():
        n = len(vertices)
        if n == 1:
            # For a single vertex, assign a default y coordinate with a slight random offset
            y_positions = [spacing / 2 + np.random.uniform(-epsilon, epsilon)]
        else:
            # Evenly space vertices over [0, spacing], adding a small epsilon offset to each
            y_positions = [i * spacing / (n - 1) + np.random.uniform(-epsilon, epsilon) for i in range(n)]
        for v, y in zip(vertices, y_positions):
            pos[v] = [layer_to_x[layer_val], y]

    # Check the overall bounding box of pos
    xs = [pos[v][0] for v in g.vertices()]
    ys = [pos[v][1] for v in g.vertices()]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width < epsilon or height < epsilon:
        raise ValueError("Layout bounding box is degenerate. Increase spacing or epsilon.")
    return pos


def bipartite_ordered_layout(
    g,
    left_val,
    right_val,
    layer_prop='layer_decoded',
    sort_left_by=lambda v: int(v),
    vertical_spacing=30.0,
    horizontal_spacing=1.0,
):
    """
    Arrange a bipartite graph so edges are as horizontal as possible:
      1) Identify the left set (layer == left_val) and the right set (layer == right_val).
      2) Sort the left set by a given key function (default: vertex id).
      3) Sort each node on the right by the average index of its neighbors on the left.
      4) Assign x=0 for the left side, x=horizontal_spacing for the right side.
         Multiply the y-index by vertical_spacing for each side.

    Parameters
    ----------
    g : graph_tool.Graph or GraphView
        The bipartite graph.
    left_val : str
        The property value used for the left side. E.g. 'layer_1'
    right_val : str
        The property value used for the right side. E.g. 'layer_2'
    layer_prop : str, optional
        Vertex property name that stores the layer. Default: 'layer_decoded'.
    sort_left_by : callable, optional
        A function used to sort the left side's vertices. Default: sorts by vertex ID.
    vertical_spacing : float, optional
        Multiplier for vertical distances. A larger value spreads nodes further vertically.
        Default is 30.0.
    horizontal_spacing : float, optional
        The x distance between the left and right columns. Default is 1.0.

    Returns
    -------
    pos : VertexPropertyMap
        A 2D coordinate property map for graph-tool, with x=0 or x=horizontal_spacing for each side
        and y determined by the sorted index times vertical_spacing.
    """

    # Separate nodes into left and right sets
    left_nodes = []
    right_nodes = []
    for v in g.vertices():
        val = g.vp[layer_prop][v]
        if val == left_val:
            left_nodes.append(v)
        elif val == right_val:
            right_nodes.append(v)

    # Sort the left side by the provided key
    left_nodes.sort(key=sort_left_by)
    # Assign an integer index to each node on the left
    left_index = {v: i for i, v in enumerate(left_nodes)}

    # For each node on the right, compute the average index of its neighbors on the left
    def avg_left_index(v):
        indices = []
        for w in v.all_neighbors():
            if w in left_index:
                indices.append(left_index[w])
        if indices:
            return sum(indices) / len(indices)
        else:
            return 0

    # Sort the right side by the average neighbor index on the left
    right_nodes.sort(key=avg_left_index)
    # Assign an integer index to each node on the right
    right_index = {v: i for i, v in enumerate(right_nodes)}

    # Create a coordinate property map
    pos = g.new_vertex_property("vector<double>")

    # For left side, x=0; for right side, x=horizontal_spacing
    # Multiply the index by vertical_spacing for the y-coordinate
    for v in left_nodes:
        pos[v] = [0.0, left_index[v] * vertical_spacing]
    for v in right_nodes:
        pos[v] = [horizontal_spacing, right_index[v] * vertical_spacing]

    return pos


def load_or_compute_layout(g, filename, override=False, inject=None):
    """
    Loads vertex layout coordinates from a TSV file if it exists (and override is False), 
    or computes/injects them and saves them to the file.

    The TSV file is expected to have the following key columns:
      - Either: layer_hash and node_id_hash (primary keys)
      - Or: v_int (if layer and node hash are not available)
      - Additionally, x and y for coordinates.

    Parameters
    ----------
    g : graph_tool.Graph
        The graph for which the layout is needed.
    filename : str
        The path to the TSV file where layout coordinates are stored or should be saved.
    override : bool, optional
        If True, recompute/inject the layout even if the file already exists and update the file. Default is False.
    inject : None, callable, or a precomputed layout, optional
        If provided, use this layout instead of computing via sfdp_layout. If callable, it should accept the graph `g`
        and return a layout (vertex property map). Otherwise, it should be a layout that maps vertices to [x, y] coordinates.

    Returns
    -------
    pos : graph_tool.VertexPropertyMap
        A vertex property map containing the 2D coordinates for each vertex.
    """
    # Determine the key type based on graph properties
    if "layer_hash" in g.vp and "node_id_hash" in g.vp:
        key_type = "hash"
    elif "v_int" in g.vp:
        key_type = "v_int"
    else:
        raise ValueError("Graph does not have the required key properties ('layer_hash' and 'node_id_hash', or 'v_int').")
    
    # Use the injected layout if provided
    if inject is not None:
        pos = inject(g) if callable(inject) else inject
        
        data = []
        for v in g.vertices():
            row = {}
            if key_type == "hash":
                row["layer_hash"] = g.vp["layer_hash"][v]
                row["node_id_hash"] = g.vp["node_id_hash"][v]
            else:
                row["v_int"] = int(g.vp["v_int"][v])
            x, y = pos[v]
            row["x"] = x
            row["y"] = y
            data.append(row)
        df = pd.DataFrame(data)
        df.to_csv(filename, sep="\t", index=False)
        print(f"Injected layout saved for {len(data)} vertices to {filename}")
    
    # If no injection is provided, try to load from file if it exists and override is False.
    elif os.path.exists(filename) and not override:
        df = pd.read_csv(filename, delimiter="\t")
        pos = g.new_vertex_property("vector<double>")
        # Determine key type based on file columns
        if "layer_hash" in df.columns and "node_id_hash" in df.columns:
            file_key = "hash"
        elif "v_int" in df.columns:
            file_key = "v_int"
        else:
            raise ValueError("TSV file does not have the required key columns.")
        
        for v in g.vertices():
            if file_key == "hash":
                lh = g.vp["layer_hash"][v]
                nid = g.vp["node_id_hash"][v]
                row = df[(df["layer_hash"] == lh) & (df["node_id_hash"] == nid)]
                if row.empty:
                    raise ValueError(f"No saved layout information found for vertex with layer_hash {lh} and node_id_hash {nid}.")
            else:
                v_int = int(g.vp["v_int"][v])
                row = df[df["v_int"] == v_int]
                if row.empty:
                    raise ValueError(f"No saved layout information found for vertex with v_int {v_int}.")
            x = float(row.iloc[0]["x"])
            y = float(row.iloc[0]["y"])
            pos[v] = [x, y]
        print(f"Loaded layout for {len(df)} vertices from {filename}")
    
    # Otherwise, compute the layout using sfdp_layout.
    else:
        pos = sfdp_layout(g)
        
        data = []
        for v in g.vertices():
            row = {}
            if key_type == "hash":
                row["layer_hash"] = g.vp["layer_hash"][v]
                row["node_id_hash"] = g.vp["node_id_hash"][v]
            else:
                row["v_int"] = int(g.vp["v_int"][v])
            x, y = pos[v]
            row["x"] = x
            row["y"] = y
            data.append(row)
        df = pd.DataFrame(data)
        df.to_csv(filename, sep="\t", index=False)
        if override:
            print(f"Override enabled: Computed and saved new layout for {len(data)} vertices to {filename}")
        else:
            print(f"Computed and saved layout for {len(data)} vertices to {filename}")
    
    return pos

def prop_to_size(g, prop, mi=1, ma=8, power=1, transform_func=None, mode='v'):
    """
    Scales a property to a specified size range with an optional power transformation and custom vectorized transformation.
    
    Parameters:
    -----------
    g : graph_tool.Graph
        The graph object.
    prop : array-like or PropertyMap
        The property values to scale. Can be a list, numpy array, or a graph-tool property map (g.vp or g.ep).
    mi : float
        Minimum size.
    ma : float
        Maximum size.
    power : float
        Power to apply for scaling.
    transform_func : callable, optional
        A function to apply to the property values before scaling. This function should support vectorized operations.
        If it doesn’t, np.vectorize will be used as a fallback.
    mode : str, optional
        Specifies whether the property is a vertex property ('v') or an edge property ('e'). Defaults to 'v'.
    
    Returns:
    --------
    size_prop : graph_tool.PropertyMap
        A property map with the scaled sizes, either a vertex or edge property map based on mode.
    """
    try:
        arr = np.array(prop, dtype=float)
    except Exception:
        arr = np.array(list(prop), dtype=float)
        
    if transform_func is not None:
        try:
            values = np.array(transform_func(arr), dtype=float)
        except Exception:
            values = np.vectorize(transform_func)(arr)
    else:
        values = arr

    min_val = np.min(values)
    max_val = np.max(values)
    if min_val == max_val:
        sizes = np.full(values.shape, mi)
    else:
        if power != 1:
            values = values ** power
        sizes = np.interp(values, [min_val, max_val], [mi, ma])
    
    if mode == 'v':
        size_prop = g.new_vertex_property("float", vals=sizes.tolist())
    elif mode == 'e':
        size_prop = g.new_edge_property("float", vals=sizes.tolist())
    else:
        raise ValueError("Mode must be either 'v' for vertex or 'e' for edge property.")
        
    return size_prop