from .core import OnionNetGraph
from graph_tool.all import Graph, GraphView, PropertyMap, graph_draw, shortest_distance
from graph_tool.topology import label_components
from collections import deque
from typing import List, Any, Union

#########################################
# Searcher: Graph Traversal & Subgraph Extraction
#########################################
class OnionNetSearcher:
    def __init__(self, core: OnionNetGraph):
        self.core = core

    def compute_on_shortest(self, source_idx: int, target_indices: List[int], inplace: bool = False, g: Graph = None, return_gv = False):
        """
        Computes and returns a Boolean vertex property 'on_shortest'
        for vertices that lie on some shortest path from the vertex at source_idx
        to any vertex in target_indices in an unweighted directed graph.
        
        If inplace is False (default), the computation is performed on a copy of the graph,
        leaving the original graph unmodified, and the result property map is returned for
        the original graph.
        If inplace is True, the original graph is modified during computation, but the modifications,
        including any additional vertices and edges, are removed at the end of the function.
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

        # Phase 1: Compute forward distances from source.
        forward_dist = shortest_distance(g_temp, source=source)

        # Phase 2: Compute reverse distances using the reversible view.
        original_reversed = g_temp.is_reversed()
        g_temp.set_reversed(True)
        art_source = g_temp.add_vertex()  # Add an artificial source vertex.

        # Create an edge property for weights (all real edges have weight 1).
        w = g_temp.new_edge_property("int")
        for e in g_temp.edges():
            w[e] = 1
        # For each target, add an edge from art_source with weight 0.
        for t in targets:
            e = g_temp.add_edge(art_source, t)
            w[e] = 0

        reverse_dist = shortest_distance(g_temp, source=art_source, weights=w)
        # Revert the reversed state.
        g_temp.set_reversed(original_reversed)

        # Phase 3: Mark vertices on some shortest path.
        target_dists = { forward_dist[t] for t in targets }
        on_shortest_temp = g_temp.new_vertex_property("bool")
        # Determine the number of original vertices (before adding the artificial vertex).
        num_orig = g.num_vertices() if not inplace else g_temp.num_vertices() - 1
        for v in g_temp.vertices():
            # Skip the artificial vertex if operating inplace
            if inplace and int(v) >= num_orig:
                continue
            on_shortest_temp[v] = False
            if forward_dist[v] == float("inf") or reverse_dist[v] == float("inf"):
                continue
            if forward_dist[v] + reverse_dist[v] in target_dists:
                on_shortest_temp[v] = True

        # Clean up the modifications if operating inplace.
        if inplace:
            g_temp.remove_vertex(art_source, fast=True)
            result_prop = on_shortest_temp
        else:
            # Map the computed values from the copy back to the original graph.
            result_prop = g.new_vertex_property("bool")
            for v in g.vertices():
                result_prop[v] = on_shortest_temp[v]
        if return_gv:
            GraphView(g, vfilt=result_prop)
        else:
            return result_prop

    def _bfs_traversal(self, seed_vertices, vfilt, efilt, mode='downstream'):
        """Perform a simple BFS to update vertex and edge filters."""
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
        g: Graph = None,  # Optionally use a graph or graphview instead, default is to use core
        **kwargs
    ) -> GraphView:
        g = g or self.core.graph

        def get_label(v):
            return g.vp[node_text_prop][v] if node_text_prop in g.vp else str(int(v))
        
        try:
            start_vertex = g.vertex(start_node_idx)
        except Exception as e:
            raise ValueError(f"Invalid start index {start_node_idx}: {e}")

        upstream_nodes = set()
        downstream_nodes = set()
        if direction in ('upstream', 'bi'):
            # For upstream, reverse the view only if needed.
            g_rev = GraphView(g, reversed=True)
            distances_up = shortest_distance(g_rev, source=start_vertex, max_dist=max_dist)
            upstream_nodes = {v for v in g.vertices() if distances_up[v] <= max_dist}
            if verbosity:
                print("Upstream nodes:",
                    [f"{int(v)} ({get_label(v)})" for v in upstream_nodes])
            if include_upstream_children and direction == 'bi':
                children = set()
                for v in upstream_nodes:
                    children.update(list(v.out_neighbours()))
                upstream_nodes |= children
        if direction in ('downstream', 'bi'):
            distances_down = shortest_distance(g, source=start_vertex, max_dist=max_dist)
            downstream_nodes = {v for v in g.vertices() if distances_down[v] <= max_dist}
            if verbosity:
                print("Downstream nodes:",
                    [f"{int(v)} ({get_label(v)})" for v in downstream_nodes])
        if direction == 'bi':
            final = upstream_nodes.union(downstream_nodes)
        elif direction == 'upstream':
            final = upstream_nodes
        elif direction == 'downstream':
            final = downstream_nodes
        else:
            raise ValueError("Invalid direction; choose 'upstream', 'downstream', or 'bi'.")

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
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        missing = [ln for ln in layer_names if ln not in self.core.layer_name_to_code]
        if missing:
            raise ValueError(f"Layer(s) {missing} do not exist.")
        codes = {self.core.layer_name_to_code[ln] for ln in layer_names}

        # Create a new boolean vertex property as the filter
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

    def view_components(self, size_threshold: int, connectivity: str = "strong") -> GraphView:
        directed = connectivity.lower() == "strong"
        comp, hist = label_components(self.core.graph, directed=directed)
        valid = {i for i, count in enumerate(hist) if count >= size_threshold}
        return GraphView(self.core.graph, vfilt=lambda v: comp[v] in valid)

    def filter_view_by_property(
        self, 
        prop_name: str, 
        target_value: Any, 
        comparison: str = "==",
        dim: str = 'v',
        prune_isolated: bool = False
    ) -> GraphView:
        """
        Filters the graph based on a vertex or edge property and returns a GraphView.
        
        Parameters:
            prop_name (str): Name of the property to filter by.
            target_value (Any): A single value or a list/set of values to compare.
            comparison (str): Comparison operator (only used if target_value is not a list/set).
                                Options: "==", "!=", "<", ">", "<=", ">=".
            dim (str): Dimension to filter on. Use 'v' for vertices (default) or 'e' for edges.
            prune_isolated (bool): If True, further filter the view to keep only those vertices
                                that have at least one incident edge in the filtered view.
        
        Returns:
            GraphView: A filtered view of the graph.
        
        Raises:
            ValueError: If the specified property doesn't exist or if an invalid dimension is provided.
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
                # Keep only vertices with at least one incident edge in the current view.
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
        Create a composite filter based on a list of filter functions.
        
        Parameters:
            filter_funcs (list): List of functions; each takes a vertex (or edge) and returns True 
                                if it should be kept.
            mode (str): "and" (default) requires all functions to return True,
                        "or" requires at least one to return True.
            type (str): 'v' for vertices, 'e' for edges.
            return_prop (bool): If True, return a new Boolean property map instead of a GraphView.
            g (Graph): The graph to operate on; defaults to self.core.graph.
                                    
        Returns:
            Either a GraphView or a PropertyMap (vertex or edge) of booleans.
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
            
    def create_bipartite_gv(self, layer1: str, layer2: str, prop_name: str = 'layer_decoded') -> GraphView:
        """
        Create a bipartite GraphView that only retains vertices whose property `prop_name`
        is either layer1 or layer2, and only keeps edges that connect vertices between the two layers.
        
        Parameters
        ----------
        layer1 : str
            The first layer value (e.g. 'swisslipids').
        layer2 : str
            The second layer value (e.g. 'sl_chebi').
        prop_name : str, optional
            The name of the vertex property to filter on (default is 'layer_decoded').
        
        Returns
        -------
        GraphView
            A filtered view of the graph containing only vertices in the given layers and only
            edges connecting vertices from layer1 to layer2. Additionally, vertices with no
            incident edges in the filtered view are removed.
        """
        g = self.core.graph

        # Only keep vertices whose 'prop_name' is either layer1 or layer2.
        initial_vfilt = lambda v: g.vp[prop_name][v] in {layer1, layer2}

        # Only keep edges that connect a vertex in layer1 with one in layer2.
        edge_filter = lambda e: (
            (g.vp[prop_name][e.source()] == layer1 and g.vp[prop_name][e.target()] == layer2) or
            (g.vp[prop_name][e.source()] == layer2 and g.vp[prop_name][e.target()] == layer1)
        )

        # Create a first GraphView applying the vertex and edge filters.
        gv = GraphView(g, vfilt=initial_vfilt, efilt=edge_filter)

        # Now define an additional vertex filter to keep only vertices that have at least one incident edge.
        vfilt_connected = lambda v: (v.out_degree() + v.in_degree()) > 0

        # Create a second, nested GraphView applying the additional vertex filter.
        gv2 = GraphView(gv, vfilt=vfilt_connected)
        return gv2