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


def draw_weight_propagation_graph(
    g: gt.Graph,
    measured_node: gt.Vertex,
    nested_properties: gt.PropertyMap,
    layout: str = 'radial',
    output: str = None,
    node_colors: Dict[gt.Vertex, str] = None,
    **kwargs
):
    """
    Draws a graph to visualize the weight propagation from a measured node. The measured
    node is highlighted in red, and node labels reflect any nested properties (e.g., reactions, categories).
    
    Parameters:
    ----------
    g : graph_tool.Graph
        The directed graph to be drawn.
    measured_node : graph_tool.Vertex
        The node of interest that will be highlighted in red.
    nested_properties : graph_tool.PropertyMap
        A property map containing potentially nested properties for each node (e.g., reactions or categories).
    layout : str, optional, default 'radial'
        The layout for positioning the vertices. Options are 'radial', 'fruchterman', or 'arf'.
    output : str or None, optional
        If specified, saves the graph to the given file name (e.g., 'graph.png'). If None, 
        the graph is displayed inline.
    node_colors : dict, optional
        A dictionary mapping each vertex to a specific color. If None, measured node is red, 
        and other nodes are blue.
    **kwargs : dict, optional
        Additional keyword arguments passed to `graph_tool.draw.graph_draw` to customize the
        visualization further (e.g., vertex_size, edge_color).
    
    Returns:
    -------
    None
    """
    # Create a vertex property for colors, based on node_colors or default
    vertex_colors = g.new_vertex_property("string")
    
    # Assign colors based on provided node_colors or default values
    for v in g.vertices():
        if node_colors and v in node_colors:
            vertex_colors[v] = node_colors[v]
        elif v == measured_node:
            vertex_colors[v] = "#ff0000"  # Red for measured node
            if layout == 'radial':  # Optional verbose feedback
                print('Measured node found and colored red!')
        else:
            vertex_colors[v] = "#729fcf"  # Blue for other nodes

    # Create node labels from the nested properties
    vertex_labels = create_node_labels(g, nested_properties)

    # Choose the layout for positioning nodes
    if layout == 'radial':
        pos = gt.radial_tree_layout(g, root=measured_node)
    elif layout == 'fruchterman':
        pos = gt.fruchterman_reingold_layout(g)
    elif layout == 'arf':
        pos = gt.arf_layout(g)
    else:
        raise ValueError("Invalid layout option. Choose 'radial', 'fruchterman', or 'arf'.")

    # Default arguments for graph_draw, can be overridden by kwargs
    default_args = {
        "vertex_text": vertex_labels,     # Node labels based on nested properties
        "vertex_font_size": 8,            # Font size for the labels
        "vertex_fill_color": vertex_colors, # Custom vertex colors
        "edge_pen_width": 1.2,            # Line width for edges
        "vertex_size": 34,                # Size of the nodes
        "output_size": (800, 600),        # Output dimensions
        "output": output                  # File name to save or None for inline
    }
    
    # Update default arguments with user-specified kwargs
    draw_args = {**default_args, **kwargs}

    # Draw the graph with custom vertex colors and node properties as labels
    gt.graph_draw(g, pos=pos, **draw_args)

    # Show the graph inline if output is None
    if output is None:
        plt.show()

    # Example usage:
    # Assuming g, measured_node, and nested_properties are defined
    # draw_weight_propagation_graph(g, measured_node=nodes[2], nested_properties=reactions, layout='radial', vertex_size=40, edge_color="#333333")


def extract_category_weights(accumulated_categories, reaction_based=False, cat_of_interest=None):
    """
    Extracts category weights from the pathlog data stored in the accumulated_categories.
    
    Parameters:
    ----------
    accumulated_categories : dict
        A dictionary containing a key 'pathlog' which is a list of dictionaries. Each dictionary contains 
        'current_category', 'node', and 'weight' for each entry.
    
    Returns:
    -------
    dict
        A dictionary where keys are categories and values are dictionaries mapping nodes to their weights.
    """
    if reaction_based==True:
        # Initialize category weights dictionary
        category_weights = defaultdict(dict)

        # Populate category weights
        for entry in accumulated_categories['pathlog']:
            category = entry['current_category']
            node = entry['node']
            weight = entry['weight']
            
            # Add node's weight to the category
            category_weights[category][node] = weight

        # Convert defaultdict to a regular dict for final output
        return dict(category_weights)
        # Example usage:
        # category_weights = extract_category_weights(accumulated_categories)
    else:
        if cat_of_interest != None:
            # # e.g. cat_of_interest = 'R2'
            # cat_of_interest_nodeweights = { # logic needed changing to avoid including upsttream inheritances
            #     item['node']: item['weight'] if (any([item==cat_of_interest for item in item['inherited_categories'].keys()]) or any([item==cat_of_interest for item in item['current_category']]) and str(item['current_origin'])in['downstream']) ^ (any([item==cat_of_interest for item in item['current_category']]) and str(item['current_origin'])in['start']) ^ (any([item==cat_of_interest for item in item['current_category']])) else 0.0 
            #     for item in accumulated_categories['pathlog']
            # }
            # print(cat_of_interest_nodeweights)
            # print({ # logic needed changing to avoid including upsttream inheritances
            #     item['node']: item if (any([item==cat_of_interest for item in item['inherited_categories'].keys()]) and str(item['current_origin'])in['downstream']) or (str(item['current_origin'])in['start']) or (any([item==cat_of_interest for item in item['current_category']])) else 0.0 
            #     for item in accumulated_categories['pathlog']
            # })
            # Assuming 'accumulated_categories' is a dictionary containing 'pathlog' as a list of entries
            cat_of_interest_nodeweights = {}

            for entry in accumulated_categories.get('pathlog', []):
                node = entry.get('node')
                weight = entry.get('weight', 0.0)
                inherited_categories = entry.get('inherited_categories', {})
                current_origin = str(entry.get('current_origin', ''))
                current_categories = entry.get('current_category', [])

                # Check if 'cat_of_interest' is in inherited_categories and 'upstream'
                has_inherited_cat = any(cat == cat_of_interest for cat in inherited_categories.keys())
                is_upstream = current_origin.lower() == 'upstream'
                
                is_downstream = current_origin.lower() == 'downstream'
                is_starter = current_origin.lower() == 'start'

                # Check if 'cat_of_interest' is in current_category
                has_current_cat = cat_of_interest == current_categories
                print(current_categories)
                print(cat_of_interest)

                print(f"Entry {node}: has_inherited_cat:{has_inherited_cat}, is_downstream:{is_downstream}, has_current_cat:{has_current_cat}\nOverall evaluation:{(has_inherited_cat and is_downstream) or (has_current_cat)}")
                # if cat_of_interest in [item for item in ]
                # Determine if the node should include its weight
                if (has_inherited_cat and is_downstream) or (has_current_cat):
                    print(f'Adding weight of {weight}')
                    cat_of_interest_nodeweights[node] = weight
                    print(f'Added weight under node: {node}')
                elif not is_upstream:
                    print('Not upstream')
                    # still need to check that it's already added, as could be the case with 2 or more properties, in which case if we didn't do this it will overwrite...
                    if node not in cat_of_interest_nodeweights.keys():
                        cat_of_interest_nodeweights[node] = 0.0
                        print(f'Setting weigth to 0 for node: {node}')
                # elif is_starter:
                #     cat_of_interest_nodeweights[node] = weight
                # elif is_upstream:
                #     print('Is upstream!!')
                #     print(inherited_categories)
                #     if cat_of_interest not in inherited_categories:
                #         cat_of_interest_nodeweights[node] = weight
                #     else:
                #         if cat_of_interest in current_categories:
                #             cat_of_interest_nodeweights[node] = weight
                # elif is_upstream or is_starter:
                #     if (cat_of_interest not in inherited_categories) or (cat_of_interest in current_categories):
                #         cat_of_interest_nodeweights[node] = weight
                #         print(f"Added upstream or starter node {node} under weight {weight}")
                    # print(inherited_categories)
                    # print(inherited_categories.keys())
                    # categories_inherited = [cat for cat in inherited_categories.keys()]
                    # for inherited_cat in categories_inherited:
                    #     print(inherited_cat)
                    #cat_of_interest_nodeweights[cat] = weight
                    #print(f"Added upstream node {cat} under weight {weight}")
                else:
                    # still need to check that it's already added, as could be the case with 2 or more properties, in which case if we didn't do this it will overwrite...
                    if node not in cat_of_interest_nodeweights.keys():
                        cat_of_interest_nodeweights[node] = 0.0
                        print(f'Final call, setting weigth to 0 for node: {node}')
            return cat_of_interest_nodeweights
            # Example output:
            # {2: 1.0,
            # 3: 0.5,
            # 5: 0.25,
            # etc...
        else:
            print('You need to specify the category of interest.')

# Printing the category weights (can be done outside the function)
def print_category_weights(category_weights):
    for category, nodes in category_weights.items():
        print(f"Category {category}:")
        for node, weight in nodes.items():
            print(f"  Node {node}: Weight {weight}")
    # Example usage:
    # print_category_weights(category_weights)




# Updated function to draw subplots
def draw_category_propagation_subplots(
    g: gt.Graph,
    measured_node: gt.Vertex,
    accumulated_categories: dict,  # Pass the accumulated_categories dict here
    layout: str = 'radial',
    categories: List[str] = None,
    output: str = None,
    **kwargs
):
    """
    Draws a set of subplots showing how individual categories propagate their weights throughout the network.
    
    Parameters:
    ----------
    g : graph_tool.Graph
        The directed graph to be drawn.
    measured_node : graph_tool.Vertex
        The node of interest that will be highlighted in red.
    accumulated_categories : dict
        A dictionary containing a key 'pathlog' which holds the list of nodes and their associated category weights.
    layout : str, optional, default 'radial'
        The layout for positioning the vertices. Options are 'radial', 'fruchterman', or 'arf'.
    categories : list of str, optional
        The list of categories to be visualized. If None, all categories in category_weights will be visualized.
    output : str or None, optional
        If specified, saves the graph to the given file name (e.g., 'graph.png'). If None, the graph is displayed inline.
    **kwargs : dict, optional
        Additional keyword arguments passed to `graph_tool.draw.graph_draw` to customize the visualization further.
    
    Returns:
    -------
    None
    """
    # Extract category weights using the function
    category_weights = extract_category_weights(accumulated_categories, reaction_based=True)

    # Use all categories if none are specified
    if categories is None:
        categories = list(category_weights.keys())
    print(f"Categories: {categories}")
    
    # Create subplots: one for each category
    num_categories = len(categories)
    #fig, axes = plt.subplots(1, num_categories, figsize=(15, 5))

    #if num_categories == 1:
    #    axes = [axes]  # Handle single subplot case
    
    # Choose the layout for positioning nodes
    if layout == 'radial':
        pos = gt.radial_tree_layout(g, root=measured_node)
    elif layout == 'fruchterman':
        pos = gt.fruchterman_reingold_layout(g)
    elif layout == 'arf':
        pos = gt.arf_layout(g)
    else:
        raise ValueError("Invalid layout option. Choose 'radial', 'fruchterman', or 'arf'.")

    graphs_output = []

    # Plot each category in a separate subplot
    for i, category in enumerate(categories):
        #ax = axes[i]
        
        # Create a vertex property for the colors based on the weights for this category
        vertex_colors = g.new_vertex_property("vector<double>")
        vertex_weights = {}
        # Create a new vertex property to store weights as strings
        vertex_text_property = g.new_vertex_property("string")
        # Use a matplotlib color map (e.g., viridis) for continuous coloring
        cmap = cm.get_cmap('viridis')  # You can choose other colormaps such as 'plasma', 'inferno', etc.
        # max_weight = max(category_weights[category].values())
        
        # Assign colors based on the weight
        # for v in g.vertices():
        #     node_index = int(v) # get the node index from the graph vertex
        #     weight = category_weights[category].get(node_index, 0) # use the correct index node
        #     normalized_weight = weight / max_weight if max_weight > 0 else 0
        #     vertex_colors[v] = [1.0 - normalized_weight, 1.0, 1.0 - normalized_weight, 1.0]  # Green for higher weights
        # Assign colors based on the weight
        extracted_category_weights = extract_category_weights(accumulated_categories, reaction_based=False, cat_of_interest=category)
        print(f'Extracted category weights: {extracted_category_weights}')
        for v in g.vertices():
            #node_index = int(v)
            #weight = category_weights.get(category, {}).get(node_index, 0)  # Access weight for the category
            node_index = int(v) # get the node index from the graph vertex
            weight = extracted_category_weights[node_index] #.get(node_index, 0)
            normalized_weight = weight if weight > 0 else 0 #/ max_weight if max_weight > 0 else 0
            #print(weight)
            #print(normalized_weight)
            #vertex_colors[v] = [1.0 - normalized_weight, 1.0, 1.0 - normalized_weight, 1.0]  # Green for higher weights
            #print(vertex_colors)
            print(f"Node {node_index}: Weight {weight} for category {category}")  # Debugging print
            vertex_weights[category] = g.new_vertex_property("vector<float>")
            color = cmap(weight)  # Returns a tuple (R, G, B, A)
            vertex_colors[v] = color  # Assign color to each vertex

            # Assign the string representation of the weights to this property
            vertex_text_property[v] = f"{weight:.2f}"  # Format the weight as a string with 2 decimal places

        # Default arguments for graph_draw, can be overridden by kwargs
        default_args = {
            "vertex_text": vertex_text_property,     # Node labels based on the category / property weigths
            "vertex_font_size": 8,            # Font size for the labels
            "vertex_fill_color": vertex_colors, # Custom vertex colors
            "edge_pen_width": 1.8,            # Line width for edges
            "vertex_size": 34,                # Size of the nodes
            "edge_marker_size":6,
            "output_size": (800, 600),        # Output dimensions
            "output": output                  # File name to save or None for inline
        }
        
        # Update default arguments with user-specified kwargs
        draw_args = {**default_args, **kwargs}
        print(draw_args['vertex_text'])
        print(f'\nNode Weighting Plot for {category}')
        individual_graph = gt.graph_draw(g, pos=pos, **draw_args) 
            # g,
            # pos=pos,
            # vertex_fill_color=vertex_colors,  # Use color map based on weights
            # edge_pen_width=1.2,
            # vertex_text=vertex_weights[category],
            # vertex_size=34,
            # output_size=(800, 600),
            # #mplfig=ax,                        # Plot into the respective subplot
            # **kwargs

        # Show the graph inline if output is None
        if output is None:
            plt.show()

        graphs_output.append(individual_graph)
        
    return graphs_output

        # Set the title for the category
        #ax.set_title(f"Category {category}")

        # Draw the graph with custom vertex colors
    #     individual_graph = gt.graph_draw(
    #         g,
    #         pos=pos,
    #         vertex_fill_color=vertex_colors,  # Use color map based on weights
    #         edge_pen_width=1.2,
    #         vertex_size=34,
    #         output_size=(800, 600),
    #         mplfig=ax,                        # Plot into the respective subplot
    #         **kwargs
    #     )
    #     return individual_graph

    # plt.tight_layout()
    
    # # Save or show the figure
    # if output:
    #     plt.savefig(output)
    # else:
    #     plt.show()


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


def get_legend(gv, prop, ordered_cats=None, verbose=False):
    """
    This function will create a legend with colours corresponding to those used in the plot.
    Property should correspond to the property category used for colouring the graph.
    The order of the items in the legend will be determined by the category order provided, if any.
    """
    categories = set(gv.vp[prop])
    # Use a predefined color map (e.g., tab10) for distinct color assignment to each category
    color_map = {cat: cm.tab10(i % 10) for i, cat in enumerate(categories)} # this should reflect the internal colour property function in lipinet
    if verbose==True:
        print(color_map)

    # Create legend elements in the specified order
    if ordered_cats != None:
        ordered_categories = ordered_cats # e.g. ['Category', 'Class', 'Species', 'Molecular subspecies', 'Structural subspecies', 'Isomeric subspecies']
        legend_elements = [Patch(facecolor=color_map[cat][:3], label=cat) for cat in ordered_categories]
    else:
        legend_elements = [Patch(facecolor=color[:3], label=category)  # Use RGB only
                            for category, color in color_map.items()]

    # Plot the legend
    plt.figure(figsize=(5, 3))
    plt.legend(handles=legend_elements, title=prop.capitalize(), loc="center", frameon=False)
    plt.axis("off")  # Hide axes for a clean legend display
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