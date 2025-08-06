from .core import OnionNetGraph
from graph_tool.all import Graph, GraphView, PropertyMap, graph_draw, shortest_distance
from graph_tool.topology import label_components
from collections import deque
from typing import List, Any, Union, Callable

from .property_manager import OnionNetPropertyManager

"""
This module defines the OnionNetSearcher class, which provides functionality for graph traversal and subgraph extraction 
within an OnionNetGraph. It includes methods for computing shortest path related properties, performing breadth-first search 
traversals, and generating filtered graph views based on various criteria.
"""

#########################################
# Searcher: Graph Traversal & Subgraph Extraction
#########################################
class OnionNetSearcher:
    def __init__(self, core: OnionNetGraph):
        """
        Initialize the OnionNetSearcher with a core OnionNetGraph instance.
        
        Parameters:
            core (OnionNetGraph): The core graph object that will be used for searching and traversal operations.
        """
        self.core = core

    def compute_on_shortest(self, source_idx: int, target_indices: List[int], inplace: bool = False, g: Graph = None, return_gv: bool = False):
        """
        Compute and return a Boolean vertex property 'on_shortest' for vertices that lie on some shortest path
        from the vertex at source_idx to any vertex in target_indices in an unweighted directed graph.
        
        The function performs the following steps:
          1. Computes forward distances from the source vertex.
          2. Computes reverse distances using a reversed graph view and an artificial source vertex.
          3. Marks vertices that lie on a shortest path if the sum of forward and reverse distances matches 
             a target's distance.
        
        Parameters:
            source_idx (int): The index of the source vertex.
            target_indices (List[int]): A list of target vertex indices.
            inplace (bool, optional): If False (default), the computation is done on a copy of the graph; if True, 
                                      the computation is performed in-place on the original graph.
            g (Graph, optional): An optional graph to operate on; defaults to self.core.graph if not provided.
            return_gv (bool, optional): If True, returns a GraphView filtered by the computed property; otherwise,
                                        returns the Boolean property map.
        
        Returns:
            Union[GraphView, PropertyMap]: A GraphView if return_gv is True, or the Boolean property map otherwise.
        
        Raises:
            ValueError: If the source index or any target index is invalid.
        """
        g = g or self.core.graph
        if not inplace:
            g_temp = g.copy()
        else:
            g_temp = g

        try:
            source = g_temp.vertex(source_idx)
        except Exception as e:
            raise ValueError(f"Invalid source index {source_idx}: {e}")
        targets = []
        for idx in target_indices:
            try:
                targets.append(g_temp.vertex(idx))
            except Exception as e:
                raise ValueError(f"Invalid target index {idx}: {e}")

        # Phase 1: Compute forward distances from the source.
        forward_dist = shortest_distance(g_temp, source=source)

        # Phase 2: Compute reverse distances using a reversed graph view.
        original_reversed = g_temp.is_reversed()
        g_temp.set_reversed(True)
        art_source = g_temp.add_vertex()  # Add an artificial source vertex.

        # Create an edge property for weights (all real edges have weight 1).
        w = g_temp.new_edge_property("int")
        for e in g_temp.edges():
            w[e] = 1
        # For each target, add an edge from the artificial source with weight 0.
        for t in targets:
            e = g_temp.add_edge(art_source, t)
            w[e] = 0

        reverse_dist = shortest_distance(g_temp, source=art_source, weights=w)
        # Revert to the original reversed state.
        g_temp.set_reversed(original_reversed)

        # Phase 3: Mark vertices on some shortest path.
        target_dists = { forward_dist[t] for t in targets }
        on_shortest_temp = g_temp.new_vertex_property("bool")
        # Determine the number of original vertices (before adding the artificial vertex).
        num_orig = g.num_vertices() if not inplace else g_temp.num_vertices() - 1
        for v in g_temp.vertices():
            # Skip the artificial vertex if operating in-place.
            if inplace and int(v) >= num_orig:
                continue
            on_shortest_temp[v] = False
            if forward_dist[v] == float("inf") or reverse_dist[v] == float("inf"):
                continue
            if forward_dist[v] + reverse_dist[v] in target_dists:
                on_shortest_temp[v] = True

        # Clean up modifications.
        if inplace:
            g_temp.remove_vertex(art_source, fast=True)
            result_prop = on_shortest_temp
        else:
            # Map computed values from the temporary graph back to the original graph.
            result_prop = g.new_vertex_property("bool")
            for v in g.vertices():
                result_prop[v] = on_shortest_temp[v]
        if return_gv:
            return GraphView(g, vfilt=result_prop)
        else:
            return result_prop

    def _bfs_traversal(self, seed_vertices, vfilt, efilt, mode='downstream'):
        """
        Perform a breadth-first search (BFS) traversal starting from the seed vertices and update the vertex 
        and edge filters accordingly.
        
        Parameters:
            seed_vertices (iterable): An iterable of starting vertices for the BFS.
            vfilt (PropertyMap): A Boolean vertex property map to be updated with visited vertices.
            efilt (PropertyMap): A Boolean edge property map to be updated with traversed edges.
            mode (str, optional): Direction of traversal; 'downstream' (default) for forward traversal or 
                                  'upstream' for reverse traversal.
        
        Raises:
            ValueError: If mode is not 'upstream' or 'downstream'.
        """
        visited = set()
        queue = deque(seed_vertices)
        while queue:
            v = queue.popleft()
            if v in visited:
                continue
            visited.add(v)
            vfilt[v] = True
            if mode == 'downstream':
                for e in v.out_edges():
                    target = e.target()
                    efilt[e] = True
                    if target not in visited:
                        queue.append(target)
            elif mode == 'upstream':
                for e in v.in_edges():
                    source = e.source()
                    efilt[e] = True
                    if source not in visited:
                        queue.append(source)
            else:
                raise ValueError("Mode must be 'upstream' or 'downstream'.")

    def search(
        self,
        start_node_idx: int = 0,
        max_dist: int = 5,
        direction: str = 'downstream',
        node_text_prop: str = 'node_label',
        show_plot: bool = True,
        include_upstream_children: bool = False,
        verbosity: bool = False,
        g: Graph = None,
        **kwargs
    ) -> GraphView:
        """
        Perform a search on the graph to extract a subgraph within a specified distance from a starting node.
        
        The search can be conducted in 'downstream', 'upstream', bidirectional ('bi'), or non-directed ('any') mode. It computes the 
        shortest distances from the starting vertex and returns a GraphView containing vertices within the specified 
        maximum distance. Optionally, the subgraph can be plotted.
        
        Parameters:
            start_node_idx (int, optional): The index of the starting vertex (default is 0).
            max_dist (int, optional): Maximum distance (in hops) from the starting vertex (default is 5).
            direction (str, optional): Direction of search; 'downstream', 'upstream', 'bi' for bidirectional, or 'any' for non-directed (default is 'downstream').
            node_text_prop (str, optional): Vertex property to use for node labels in the plot (default is 'node_label').
            show_plot (bool, optional): If True, displays a plot of the filtered subgraph (default is True).
            include_upstream_children (bool, optional): For bidirectional search, if True, include additional upstream children (default is False).
            verbosity (bool, optional): If True, prints detailed information during the search process (default is False).
            g (Graph, optional): An optional graph to operate on; defaults to self.core.graph if not provided.
            **kwargs: Additional keyword arguments passed to graph_draw for plotting.
        
        Returns:
            GraphView: A filtered view of the graph containing vertices within the specified distance from the start vertex.
        
        Raises:
            ValueError: If the starting vertex index is invalid or if an invalid search direction is specified.
        """
        g = g or self.core.graph

        def get_label(v):
            return g.vp[node_text_prop][v] if node_text_prop in g.vp else str(int(v))
        
        try:
            start_vertex = g.vertex(start_node_idx)
        except Exception as e:
            raise ValueError(f"Invalid start index {start_node_idx}: {e}")

        if direction == 'any':
            # create an undirected view of g
            g_und = GraphView(g, directed=False)
            # compute shortest‐path distances on undirected graph
            distances = shortest_distance(g_und,
                                        source=start_vertex,
                                        max_dist=max_dist)
            final = {v for v in g_und.vertices() if distances[v] <= max_dist}
            if verbosity:
                print("All-directions nodes:",
                    [f"{int(v)} ({get_label(v)})" for v in final])

            # wrap into GraphView, plot, and return
            final_indices = {int(v) for v in final}
            result = GraphView(g, vfilt=lambda v: int(v) in final_indices)
            print(f"Filtered graph contains {result.num_vertices()} vertices and {result.num_edges()} edges.")
            if show_plot:
                if node_text_prop in g.vp:
                    vertex_text = g.vp[node_text_prop]
                else:
                    vertex_text = g.new_vertex_property('string')
                    for v in result.vertices():
                        vertex_text[v] = str(int(v))
                graph_draw(result, vertex_text=vertex_text, **kwargs)
            return result

        upstream_nodes = set()
        downstream_nodes = set()
        if direction in ('upstream', 'bi'):
            # Create a reversed graph view for upstream search.
            g_rev = GraphView(g, reversed=True)
            distances_up = shortest_distance(g_rev, source=start_vertex, max_dist=max_dist)
            upstream_nodes = {v for v in g.vertices() if distances_up[v] <= max_dist}
            if verbosity:
                print("Upstream nodes:", [f"{int(v)} ({get_label(v)})" for v in upstream_nodes])
            if include_upstream_children and direction == 'bi':
                children = set()
                for v in upstream_nodes:
                    children.update(list(v.out_neighbours()))
                upstream_nodes |= children
        if direction in ('downstream', 'bi'):
            distances_down = shortest_distance(g, source=start_vertex, max_dist=max_dist)
            downstream_nodes = {v for v in g.vertices() if distances_down[v] <= max_dist}
            if verbosity:
                print("Downstream nodes:", [f"{int(v)} ({get_label(v)})" for v in downstream_nodes])
        if direction == 'bi':
            final = upstream_nodes.union(downstream_nodes)
        elif direction == 'upstream':
            final = upstream_nodes
        elif direction == 'downstream':
            final = downstream_nodes
        else:
            raise ValueError("Invalid direction; choose 'upstream', 'downstream', 'bi', or 'any'.")

        final_indices = {int(v) for v in final}
        result = GraphView(g, vfilt=lambda v: int(v) in final_indices)
        print(f"Filtered graph contains {result.num_vertices()} vertices and {result.num_edges()} edges.")
        if show_plot:
            if node_text_prop in g.vp:
                vertex_text = g.vp[node_text_prop]
            else:
                vertex_text = g.new_vertex_property('string')
                for v in result.vertices():
                    vertex_text[v] = str(int(v))
            graph_draw(result, vertex_text=vertex_text, **kwargs)
        return result

    def view_layers(
        self, 
        layer_names: Union[List[str], str],
        return_filter: bool = False,
        copy_gv: bool = False
    ) -> Union[GraphView, PropertyMap]:
        """
        Generate a GraphView filtered by the specified layer names.
        
        Parameters:
            layer_names (Union[List[str], str]): A single layer name or a list of layer names to filter vertices by.
            return_filter (bool, optional): If True, returns the Boolean vertex property used for filtering instead of a GraphView.
            copy_gv (bool, optional): If True, returns a new Graph object constructed from the GraphView.
        
        Returns:
            Union[GraphView, PropertyMap]: The filtered GraphView or Boolean property map based on the layer filter.
        
        Raises:
            ValueError: If any specified layer name does not exist.
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        missing = [ln for ln in layer_names if ln not in self.core.layer_name_to_code]
        if missing:
            raise ValueError(f"Layer(s) {missing} do not exist.")
        codes = {self.core.layer_name_to_code[ln] for ln in layer_names}

        # Create a Boolean vertex property filter based on the specified layer codes.
        vfilt = self.core.graph.new_vertex_property('bool')
        for v in self.core.graph.vertices():
            vfilt[v] = self.core.graph.vp['layer_hash'][v] in codes

        if return_filter:
            return vfilt
        else:
            if copy_gv:
                return Graph(GraphView(self.core.graph, vfilt=vfilt))
            else:
                return GraphView(self.core.graph, vfilt=vfilt)

    def view_components(self, size_threshold: int, connectivity: str = "strong", g: Graph = None) -> GraphView:
        """
        Create a GraphView that shows connected components of the graph with a minimum size.
        
        Parameters:
            size_threshold (int): The minimum number of vertices a component must have to be included.
            connectivity (str, optional): 'strong' for strongly connected components, otherwise weakly connected (default is "strong").
            g (Graph, optional): The graph to operate on; defaults to self.core.graph.
        
        Returns:
            GraphView: A view of the graph showing only components that meet the size threshold.
        """
        g = g or self.core.graph
        directed = connectivity.lower() == "strong"
        comp, hist = label_components(g, directed=directed)
        valid = {i for i, count in enumerate(hist) if count >= size_threshold}
        return GraphView(g, vfilt=lambda v: comp[v] in valid)

    def filter_view_by_property(
        self, 
        prop_name: str, 
        target_value: Any, 
        comparison: str = "==",
        dim: str = 'v',
        prune_isolated: bool = False
    ) -> GraphView:
        """
        Filter the graph based on a specified vertex or edge property and return a GraphView.
        
        Parameters:
            prop_name (str): The property name to filter by.
            target_value (Any): The value or set of values to compare against.
            comparison (str, optional): Comparison operator (default "=="). Options: "==", "!=", "<", ">", "<=", ">=".
            dim (str, optional): Dimension to filter on; 'v' for vertices (default) or 'e' for edges.
            prune_isolated (bool, optional): If True, further filters the view to retain only vertices with at least one incident edge.
        
        Returns:
            GraphView: A filtered view of the graph based on the property filter.
        
        Raises:
            ValueError: If the property does not exist or an invalid dimension is provided.
        """
        import operator
        ops = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
               ">": operator.gt, "<=": operator.le, ">=": operator.ge}

        if dim == 'v':
            if prop_name not in self.core.graph.vp:
                raise ValueError(f"Vertex property '{prop_name}' does not exist.")
            prop = self.core.graph.vp[prop_name]
            if isinstance(target_value, (list, tuple, set)):
                filt_func = lambda v: prop[v] in target_value
            else:
                if comparison not in ops:
                    raise ValueError(f"Invalid comparison operator '{comparison}'.")
                cmp_op = ops[comparison]
                filt_func = lambda v: cmp_op(prop[v], target_value)
            gv = GraphView(self.core.graph, vfilt=filt_func)
            if prune_isolated:
                gv = GraphView(gv, vfilt=lambda v: (v.out_degree() + v.in_degree()) > 0)
            return gv

        elif dim == 'e':
            if prop_name not in self.core.graph.ep:
                raise ValueError(f"Edge property '{prop_name}' does not exist.")
            prop = self.core.graph.ep[prop_name]
            if isinstance(target_value, (list, tuple, set)):
                filt_func = lambda e: prop[e] in target_value
            else:
                if comparison not in ops:
                    raise ValueError(f"Invalid comparison operator '{comparison}'.")
                cmp_op = ops[comparison]
                filt_func = lambda e: cmp_op(prop[e], target_value)
            gv = GraphView(self.core.graph, efilt=filt_func)
            if prune_isolated:
                # Instead of gv.degree(v), use the sum of out_degree and in_degree.
                gv = GraphView(gv, vfilt=lambda v: (v.out_degree() + v.in_degree()) > 0)
            return gv

        else:
            raise ValueError("Dimension must be 'v' (vertex) or 'e' (edge).")
    
    def compose_filters(self, filter_funcs, mode="and", type='v', return_prop: bool = False, g: Graph = None):
        """
        Create a composite filter from a list of individual filter functions.
        
        Parameters:
            filter_funcs (list): A list of functions, each accepting a vertex (or edge) and returning True if it should be kept.
            mode (str, optional): Logical combination mode; "and" (default) requires all functions to return True, "or" requires at least one.
            type (str, optional): The dimension of filtering; 'v' for vertices (default) or 'e' for edges.
            return_prop (bool, optional): If True, returns a new Boolean property map instead of a GraphView.
            g (Graph, optional): The graph to operate on; defaults to self.core.graph.
        
        Returns:
            Union[GraphView, PropertyMap]: A composite filter represented as a GraphView or a Boolean property map.
        
        Raises:
            ValueError: If an invalid mode or type is specified.
        """
        g = g or self.core.graph

        def composite(item):
            if mode == "and":
                return all(f(item) for f in filter_funcs)
            elif mode == "or":
                return any(f(item) for f in filter_funcs)
            else:
                raise ValueError("mode must be 'and' or 'or'")

        if return_prop:
            if type == 'v':
                new_prop = g.new_vertex_property("bool")
                for v in g.vertices():
                    new_prop[v] = composite(v)
                return new_prop
            elif type == 'e':
                new_prop = g.new_edge_property("bool")
                for e in g.edges():
                    new_prop[e] = composite(e)
                return new_prop
            else:
                raise ValueError("must specify either 'v' or 'e' as type")
        else:
            # Return a GraphView using the composite filter.
            if type == 'v':
                return GraphView(g, vfilt=composite)
            elif type == 'e':
                return GraphView(g, efilt=composite)
            else:
                raise ValueError("must specify either 'v' or 'e' as type")


    def filter_edges(self,
                     predicate: Callable, 
                     return_view: bool = True
                     ) -> GraphView:
        """
        Keep only those edges for which predicate(e) is True,
        then prune any isolated vertices.

        Parameters
        ----------
        predicate : Callable
            A function taking a graph-tool Edge and returning True to keep it.
        return_view : bool
            If True, returns a GraphView; if False, returns the raw edge‐bool PropertyMap.

        Returns
        -------
        GraphView or PropertyMap
        """
        g = self.core.graph        
        efilt = g.new_edge_property("bool")
        for e in g.edges():
            efilt[e] = bool(predicate(e))
        # note that this method above is safer than efilt.a = [predicate(e) for e in g.edges()] which doesn't gaurantee edge identity

        if not return_view:
            return efilt

        gv_edges = GraphView(g, efilt=efilt)
        return self._prune_isolated(gv_edges)
    

    def _prune_isolated(self, gv_edges):
        """
        Given a GraphView filtered on edges, drop any vertices
        that now have degree zero in that view, using vectorized assignment.
        """
        g = gv_edges.graph if hasattr(gv_edges, 'graph') else self.core.graph
        # compute degree in filtered view and get its array
        deg_map = gv_edges.degree_property_map('total')
        deg_arr = deg_map.a
        # build boolean filter: True for vertices with degree > 0
        vfilt = g.new_vertex_property('bool')
        vfilt.a = deg_arr > 0
        return GraphView(gv_edges, vfilt=vfilt)
    

    def filter_edges_between_categories(
        self,
        source_label: str,
        target_label: str,
        mode: str = "forward"
    ) -> GraphView:
        """
        Keep only those edges whose source‐vertex layer == source_label
        and whose target‐vertex layer == target_label (direction 'forward'),
        or the reverse, or both, then prune any isolates.

        mode : {'forward','reverse','both'}
          - 'forward':  source→target only  
          - 'reverse': target→source only  
          - 'both':    either direction 
        """
        g = self.core.graph

        try:
            src_code = self.core.layer_name_to_code[source_label]
            tgt_code = self.core.layer_name_to_code[target_label]
        except KeyError as e:
            raise KeyError(f"Unknown layer name: {e.args[0]}")

        lh = g.vp['layer_hash']

        if mode == "forward":
            pred = lambda e: (
                lh[e.source()] == src_code
                and lh[e.target()] == tgt_code
            )
        elif mode == "reverse":
            pred = lambda e: (
                lh[e.source()] == tgt_code
                and lh[e.target()] == src_code
            )
        elif mode == "both":
            pred = lambda e: (
                (lh[e.source()] == src_code and lh[e.target()] == tgt_code)
                or
                (lh[e.source()] == tgt_code and lh[e.target()] == src_code)
            )
        else:
            raise ValueError(f"mode must be 'forward', 'reverse' or 'both', not {mode!r}")

        return self.filter_edges(pred)


    def create_bipartite_gv(
        self,
        layer1: str,
        layer2: str,
        prop_name: str = "layer_decoded"
    ) -> GraphView:
        """
        Bipartite view: all edges crossing between layer1 and layer2,
        in either direction, then drop isolates.
        """
        # if the user wants to filter by a string‐decoded prop, build it first
        g = self.core.graph
        if prop_name not in g.vp:
            if prop_name == "layer_decoded":
                vp = g.new_vertex_property("string")
                for v in g.vertices():
                    code = int(g.vp['layer_hash'][v])
                    vp[v] = self.core.layer_code_to_name.get(code, f"Unknown({code})")
                g.vp[prop_name] = vp
            else:
                raise KeyError(f"Vertex property '{prop_name}' does not exist.")

        # Now just treat it as “both”‐direction filter on the integer layer‐hash:
        return self.filter_edges_between_categories(
            source_label=layer1,
            target_label=layer2,
            mode="both"
        )
    