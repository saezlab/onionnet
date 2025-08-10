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

from itertools import zip_longest
from graph_tool.all import Graph, GraphView

import warnings

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
    zero_centred=False,
    transparency=1.0
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
    - transparency (float): Transparency level for the colors (0.0 to 1.0). Default is 1.0 (fully opaque). 

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
                col = custom_color_dict[value]
                v_color[v] = tuple(col[:3]) + (transparency,)
            else:
                raise ValueError(f"Value '{value}' not found in custom_color_dict.")
        if generate_legend:
            legend = {k: tuple(v[:3]) + (transparency,) for k, v in custom_color_dict.items()}

    # Handle colors with custom colormap or default colormap
    elif method == "categorical":
        categories = sorted(set(g.vp[prop_name]))
        colormap = custom_colormap or cm.tab10
        colormap_len = len(colormap.colors)
        color_map = {cat: colormap(i % colormap_len) for i, cat in enumerate(categories)}
        for v in g.vertices():
            category = g.vp[prop_name][v]
            v_color[v] = color_map[category][:3] + (transparency,)
        if generate_legend:
            legend = {cat: color_map[cat][:3] + (transparency,) for cat in categories}

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
            v_color[v] = scalar_map.to_rgba(value)[:3] + (transparency,)
        if generate_legend:
            legend = {
              "min_col": scalar_map.to_rgba(min_val)[:3] + (transparency,),
              "max_col": scalar_map.to_rgba(max_val)[:3] + (transparency,),
              "min_val": min_val,
              "max_val": max_val
            }

    elif method == "boolean":
        for v in g.vertices():
            value = g.vp[prop_name][v]
            v_color[v] = (1.0, 0.0, 0.0, transparency) if bool(value) else (0.5, 0.5, 0.5, transparency)
        if generate_legend:
            legend = {"True": (1.0, 0.0, 0.0, transparency), "False": (0.5, 0.5, 0.5, transparency)}

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


def add_halos_to_nodes(
    g,
    nodes,
    colors=None,
    default_color=(1.0, 1.0, 0.0, 0.6),
    prop_name_halo="v_halo",
    prop_name_color="v_halo_color",
):
    """
    Create per-vertex halo + halo-color property maps for one or more nodes.

    Parameters
    ----------
    g : Graph or GraphView
        The graph you will draw.
    nodes : sequence of Vertex or int
        Vertices to highlight. (If you pass ints, they are treated as vertex indices.)
    colors : sequence of RGBA tuples, optional
        Per-node halo colors; if fewer than nodes, `default_color` is used for the rest.
    default_color : tuple
        Fallback RGBA.
    prop_name_halo : str
        Name to assign the boolean halo map into g.vp.
    prop_name_color : str
        Name to assign the vector<double> color map into g.vp.

    Returns
    -------
    dict
        {"v_halo": halo_bool_map, "v_halo_color": halo_color_map}
    """
    # Use the underlying base graph so property arrays line up correctly
    base = g

    v_halo       = base.new_vertex_property("bool")
    v_halo_color = base.new_vertex_property("vector<double>")

    # default: no halos
    v_halo.a[:] = False

    # Helper to coerce ints/Vertices to a Vertex on the *base* graph
    def _as_vertex(x):
        try:
            return base.vertex(int(x))
        except Exception:
            # assume it's already a Vertex from this graph
            return x

    # Assign per-node flags + colors
    for node, color in zip_longest(nodes, colors or [], fillvalue=default_color):
        v = _as_vertex(node)
        v_halo[v] = True
        v_halo_color[v] = color

    # Attach to the base graph’s vp (and also to the view’s vp for convenience)
    base.vp[prop_name_halo]  = v_halo
    base.vp[prop_name_color] = v_halo_color
    if isinstance(g, GraphView):
        g.vp[prop_name_halo]  = v_halo
        g.vp[prop_name_color] = v_halo_color

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
            # Create a Figure + Axes so colorbar() knows where to draw
            fig, ax = plt.subplots(figsize=(6, 1))
            # Attach the colorbar to that Axes
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')
            # (Optionally hide the empty image Axes if you just want the bar)
            ax.remove()
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
            # Create a Figure + Axes so colorbar() knows where to draw
            fig, ax = plt.subplots(figsize=(6, 1))
            # Attach the colorbar to that Axes
            cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')
            # (Optionally hide the empty image Axes if you just want the bar)
            ax.remove()
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
        categories = sorted(set(g.ep[prop_name]))
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
    Load or compute a 2D layout for g, keyed by either
    ('layer_decoded','node_id_decoded') if available, else
    ('layer_hash','node_id_hash').  Round-trips whatever keys you have.
    """
    # --- 1) Validate/choose key props ---
    # Preferred human-readable keys:
    use_decoded = ("layer_decoded" in g.vp) and ("node_id_decoded" in g.vp)
    # Fallback keys must always exist:
    if not use_decoded:
        if "layer_hash" not in g.vp or "node_id_hash" not in g.vp:
            raise ValueError("Graph must have either decoded props or both 'layer_hash' & 'node_id_hash'.")
    key_cols = (("layer_decoded","node_id_decoded") if use_decoded
                else ("layer_hash","node_id_hash"))

    # helper to write out a DataFrame
    def _write_df(pos):
        rows = []
        for v in g.vertices():
            row = {"x": pos[v][0], "y": pos[v][1]}
            # add whichever keys we chose
            for col in key_cols:
                # cast to Python type for CSV
                val = g.vp[col][v]
                row[col] = int(val) if isinstance(val, (int,)) else val
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(filename, sep="\t", index=False)

    # --- 2) injection branch ---
    if inject is not None:
        pos = inject(g) if callable(inject) else inject
        _write_df(pos)
        print(f"[inject] Saved layout for {len(list(g.vertices()))} vertices → {filename}")
        return pos

    # --- 3) load-from-disk if possible ---
    if os.path.exists(filename) and not override:
        df = pd.read_csv(filename, sep="\t")
        # determine which key set the file has
        if all(c in df.columns for c in ("layer_decoded","node_id_decoded")):
            file_keys = ("layer_decoded","node_id_decoded")
        elif all(c in df.columns for c in ("layer_hash","node_id_hash")):
            file_keys = ("layer_hash","node_id_hash")
        else:
            raise ValueError("TSV missing both decoded and hash key columns.")

        # build lookup from file
        lookup = {
            (row[file_keys[0]], row[file_keys[1]]): row
            for _, row in df.iterrows()
        }
        # warn about mismatches
        graph_keys = set()
        for v in g.vertices():
            key = tuple(g.vp[c][v] for c in file_keys)
            graph_keys.add(key)
        file_keyset = set(lookup.keys())
        extra_in_file  = file_keyset - graph_keys
        extra_in_graph = graph_keys - file_keyset
        if extra_in_file:
            warnings.warn(f"{len(extra_in_file)} keys in TSV not in graph: {extra_in_file}")
        if extra_in_graph:
            warnings.warn(f"{len(extra_in_graph)} graph vertices missing in TSV: {extra_in_graph}")

        # build pos
        pos = g.new_vertex_property("vector<double>")
        for v in g.vertices():
            key = tuple(g.vp[c][v] for c in file_keys)
            if key not in lookup:
                raise ValueError(f"No layout in TSV for vertex {key}")
            row = lookup[key]
            pos[v] = (float(row.x), float(row.y))

        print(f"[load]   Loaded layout for {len(df)} rows from {filename}")
        return pos

    # --- 4) compute new layout ---
    pos = sfdp_layout(g)
    _write_df(pos)
    verb = "Overrode" if override else "Computed"
    print(f"[{verb}] Saved layout for {len(list(g.vertices()))} vertices → {filename}")
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