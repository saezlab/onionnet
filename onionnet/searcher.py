from .core import OnionNetGraph
from graph_tool.all import Graph, GraphView, PropertyMap, graph_draw, shortest_distance
from graph_tool.topology import label_components
from collections import deque
from typing import List, Any, Union, Callable

from .property_manager import OnionNetPropertyManager

import numpy as np

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
        self.pm = OnionNetPropertyManager(core)

    def _coerce_to_idx(self, x):
        """Accept int/Vertex or (layer_name, node_id_str) and return int index."""
        # int or graph-tool Vertex?
        try:
            return int(x)
        except Exception:
            pass
        # (layer_name, node_id_str)?
        if isinstance(x, tuple) and len(x) == 2:
            v = self.pm.get_vertex_by_name_tuple(layer_name=x[0], node_id_str=x[1])
            if v is None:
                raise ValueError(f"Vertex {x!r} not found.")
            return int(v)
        raise TypeError("source/targets must be int/Vertex or (layer_name, node_id_str) tuple.")

    def compute_on_shortest(self, source, targets, return_gv: bool = True, inplace: bool = True, g: Graph = None):
        """
        Quickly mark nodes on *any* shortest path from a single source to one or more targets
        in a large directed (unweighted) graph, without copying the graph or adding vertices.

        1. Coerce source / targets to integer vertex indices.
        2. Run one forward BFS from the source.
        3. Run one reverse-view BFS *per target* (or single if only one).
        4. Mark v “on_shortest” if forward_dist[v] + reverse_dist[v] == dist_to_target.
        5. Optionally return a GraphView filtered by that boolean map.

        Complexity: O((V+E) + T*(V+E)) where T = len(target_indices).  
        For a single target it's just two BFS passes: ~O(V+E).

        Examples:
        ### With layer names and node IDs:
        >>> searcher.compute_on_shortest(source=(layer_name, node_id_str), targets=[(layer_name, node_id_str), ...])
        e.g.
        >>> searcher.compute_on_shortest(source=('layer1', 'nodeA'), targets=[('layer2', 'nodeB'), ('layer3', 'nodeC')])
        ### Or with integer indices:
        >>> searcher.compute_on_shortest(source=vertex_index, targets=[vertex_index, ...])
        e.g.
        >>> searcher.compute_on_shortest(source=42, targets=[43, 44])
        """
        g = g or self.core.graph
        if not inplace:
            g = g.copy()

        # 1) coerce any labels or Vertex → int, else error
        # resolve inputs via PropertyManager
        source_idx = self._coerce_to_idx(source)
        target_indices = [self._coerce_to_idx(t) for t in (targets if isinstance(targets, (list, tuple)) else [targets])]

        # 2) forward BFS from source
        forward_dist = shortest_distance(g, source=g.vertex(source_idx), directed=True)

        # 3) reverse‐view BFS back from each target
        g_rev = GraphView(g, directed=True)
        g_rev.set_reversed(True)

        inf = float("inf")
        if len(target_indices) == 1:
            reverse_dist = shortest_distance(g_rev, source=g_rev.vertex(target_indices[0]), directed=True)
        else:
            rev_min = g.new_vertex_property("double")
            rev_min.a[:] = inf
            for t in target_indices:
                d = shortest_distance(g_rev, source=g_rev.vertex(t), directed=True)
                rev_min.a = np.minimum(rev_min.a, d.a)
            reverse_dist = rev_min

        # 4) collect the target‐distance set
        dist_targets = { forward_dist[g.vertex(t)] for t in target_indices }

        # 5) build the boolean map
        on_sp = g.new_vertex_property("bool")
        for v in g.vertices():
            d1, d2 = forward_dist[v], reverse_dist[v]
            if d1 < inf and d2 < inf and (d1 + d2) in dist_targets:
                on_sp[v] = True

        # 6) return either the raw prop or a GraphView
        return GraphView(g, vfilt=on_sp) if return_gv else on_sp

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
        Filter edges by their endpoint layers and return a pruned GraphView.

        This method selects edges whose source vertex's layer matches `source_label`
        and whose target vertex's layer matches `target_label`, according to the
        integer layer codes stored in the graph's 'layer_hash' vertex property.
        You can choose one of three modes:

        - forward: keep edges where source→target
        - reverse: keep edges where target→source
        - both:    keep edges in either direction

        After filtering, any vertices that become isolated (no remaining incident edges)
        are automatically pruned.

        Vectorized to:
        - look up the integer codes for source_label and target_label
        - build a NumPy mask of edges whose (src_layer, tgt_layer) matches
            one of the allowed pairs for forward/reverse/both
        - return a pruned GraphView

        Parameters
        ----------
        source_label : str
            Human-readable name of the layer for edge sources. Must exist in
            self.core.layer_name_to_code, or a KeyError is raised.
        target_label : str
            Human-readable name of the layer for edge targets. Must exist in
            self.core.layer_name_to_code, or a KeyError is raised.
        mode : {'forward','reverse','both'}, optional
            Which direction(s) to keep:
            - 'forward': source→target only (default)
            - 'reverse': target→source only
            - 'both':    both directions

        Returns
        -------
        GraphView
            A filtered view of the underlying graph containing only the selected edges,
            with any isolated vertices removed.

        Raises
        ------
        KeyError
            If source_label or target_label is not found in the layer-name mapping.
        ValueError
            If mode is not one of 'forward', 'reverse', or 'both'.
        """
        g = self.core.graph

        # 1) map human layer names to int codes
        try:
            c1 = self.core.layer_name_to_code[source_label]
            c2 = self.core.layer_name_to_code[target_label]
        except KeyError as e:
            raise KeyError(f"Unknown layer name: {e.args[0]}")

        # 2) pull out vertex-layer array and edges *with internal edge index*
        lh_arr   = g.vp['layer_hash'].a
        edges_tbl = g.get_edges([g.edge_index])   # columns: [src, tgt, eidx]
        src_idx  = edges_tbl[:, 0]
        tgt_idx  = edges_tbl[:, 1]
        e_idx    = edges_tbl[:, 2].astype(int)    # internal edge indices

        # 3) build the boolean mask row-aligned to edges_tbl
        if mode == "forward":
            mask = (lh_arr[src_idx] == c1) & (lh_arr[tgt_idx] == c2)
        elif mode == "reverse":
            mask = (lh_arr[src_idx] == c2) & (lh_arr[tgt_idx] == c1)
        elif mode == "both":
            mask = ((lh_arr[src_idx] == c1) & (lh_arr[tgt_idx] == c2)) | \
                ((lh_arr[src_idx] == c2) & (lh_arr[tgt_idx] == c1))
        else:
            raise ValueError(f"mode must be 'forward','reverse' or 'both', not {mode!r}")

        # 4) write mask into efilt *using* the internal edge indices
        efilt = g.new_edge_property("bool")
        efilt.a = np.zeros(g.num_edges(), dtype=bool)
        efilt.a[e_idx] = mask

        gv = GraphView(g, efilt=efilt)
        return self._prune_isolated(gv)


    def create_bipartite_gv(self, layer1: str, layer2: str, prop_name: str = "layer_decoded") -> GraphView:
        # Back-compat note: prop_name is ignored; layer_decoded is not required anymore.
        # You can optionally warn here if you like.
        # warnings.warn("create_bipartite_gv is a thin wrapper. Use filter_edges_between_categories(..., mode='both').",
        #               DeprecationWarning)
        if prop_name != "layer_decoded":
            raise KeyError("prop_name is ignored now; only 'layer_decoded' was ever supported.")
        return self.filter_edges_between_categories(layer1, layer2, mode="both")