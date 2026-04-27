"""
Stage 3: Graph sparsification — find the minimal keyframe set.

Two-phase approach:
  1. Coverage-based greedy sparsification (fast)
  2. Evaluation-based refinement (iterative, optional)

Usage:
    python sparsify_graph.py --scene skokloster-castle --distance_threshold 3.0
    # Multiple thresholds at once:
    python sparsify_graph.py --scene skokloster-castle --distance_thresholds 2.0 3.0 5.0
"""

import argparse
import json
import os
import sys

import networkx as nx
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def coverage_sparsify(G: nx.Graph, distance_threshold: float) -> list:
    """Greedy set-cover sparsification.

    Each node "covers" its neighbors within `distance_threshold` NoMaD distance.
    Greedily pick the node covering the most uncovered neighbors until all
    nodes are covered (every node is either retained or adjacent to a retained node).

    Returns list of retained node IDs.
    """
    # Build coverage map: node -> set of neighbors within threshold
    coverage = {}
    for node in G.nodes():
        covered = {node}
        for neighbor in G.neighbors(node):
            edge_data = G.edges[node, neighbor]
            if edge_data["nomad_distance"] <= distance_threshold:
                covered.add(neighbor)
        coverage[node] = covered

    uncovered = set(G.nodes())
    retained = []

    while uncovered:
        # Pick the node that covers the most uncovered nodes
        best_node = max(coverage, key=lambda n: len(coverage[n] & uncovered))
        best_cover = coverage[best_node] & uncovered

        if not best_cover:
            # No single node covers any remaining — add all remaining
            retained.extend(sorted(uncovered))
            break

        retained.append(best_node)
        uncovered -= best_cover

    return retained


def refinement_sparsify(G: nx.Graph, retained: list,
                        max_removal_fraction: float = 0.5) -> list:
    """Iterative refinement: try removing nodes and keep only essential ones.

    For each candidate removal, check if the remaining graph is still
    connected (all retained nodes reachable from each other). Remove nodes
    that are "redundant" (their removal doesn't disconnect the graph and
    their neighbors are still covered by other retained nodes).

    This is a connectivity-based approximation. For full evaluation-based
    refinement, see eval_habitat_nomad.py.
    """
    retained_set = set(retained)
    retained_list = list(retained)

    # Build subgraph of retained nodes
    H = G.subgraph(retained_set).copy()

    # Sort by degree (lowest first) — low-degree nodes are more likely redundant
    nodes_by_degree = sorted(H.nodes(), key=lambda n: H.degree(n))

    removed = 0
    max_removals = int(len(retained_list) * max_removal_fraction)

    for node in nodes_by_degree:
        if removed >= max_removals:
            break

        # Try removing this node
        test_graph = H.copy()
        test_graph.remove_node(node)

        # Check: is every remaining node still connected to at least
        # one other node within NoMaD distance threshold? And is the
        # graph still connected?
        if nx.is_connected(test_graph):
            # Also check: all original non-retained nodes are still
            # covered by at least one retained neighbor
            neighbors_in_G = set(G.neighbors(node))
            retained_neighbors = neighbors_in_G & retained_set - {node}
            covered_by_others = set()
            for rn in retained_neighbors:
                covered_by_others.add(rn)
                for nn in G.neighbors(rn):
                    covered_by_others.add(nn)

            # All G nodes that were only covered by this node
            uncovered_by_removal = neighbors_in_G - covered_by_others - {node}
            if not uncovered_by_removal:
                H.remove_node(node)
                retained_set.remove(node)
                removed += 1

    return sorted(retained_set)


def main():
    parser = argparse.ArgumentParser(description="Sparsify topological graph")
    parser.add_argument("--scene", type=str, default="skokloster-castle")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "corl-2026"))
    parser.add_argument("--distance_threshold", type=float, default=3.0,
                        help="NoMaD distance threshold for coverage")
    parser.add_argument("--distance_thresholds", type=float, nargs="+", default=None,
                        help="Multiple thresholds to evaluate (overrides --distance_threshold)")
    parser.add_argument("--refine", action="store_true",
                        help="Apply connectivity-based refinement after greedy sparsification")
    parser.add_argument("--max_removal_fraction", type=float, default=0.3,
                        help="Max fraction of nodes to remove during refinement")
    args = parser.parse_args()

    # Load graph from Stage 2
    graph_path = os.path.join(args.data_dir, args.scene, "graph.json")
    if not os.path.exists(graph_path):
        print(f"Error: graph not found at {graph_path}")
        print("Run build_topo_graph.py first.")
        sys.exit(1)

    with open(graph_path) as f:
        graph_data = json.load(f)
    G = nx.node_link_graph(graph_data, directed=True)

    # Convert to undirected for sparsification
    if G.is_directed():
        G = G.to_undirected()

    total_nodes = G.number_of_nodes()
    print(f"Loaded graph: {total_nodes} nodes, {G.number_of_edges()} edges")

    # Determine thresholds to evaluate
    thresholds = (args.distance_thresholds if args.distance_thresholds
                 else [args.distance_threshold])

    results = {}
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        retained = coverage_sparsify(G, threshold)
        coverage_ratio = len(retained) / total_nodes * 100
        print(f"  Coverage sparsification: {len(retained)}/{total_nodes} "
              f"nodes retained ({coverage_ratio:.1f}%)")

        if args.refine:
            refined = refinement_sparsify(G, retained,
                                          max_removal_fraction=args.max_removal_fraction)
            refine_ratio = len(refined) / total_nodes * 100
            print(f"  After refinement: {len(refined)}/{total_nodes} "
                  f"nodes retained ({refine_ratio:.1f}%)")
            retained = refined

        # Compute stats on the sparse subgraph
        H = G.subgraph(retained).copy()
        if H.number_of_edges() > 0:
            degrees = [d for _, d in H.degree()]
            nomad_dists = [d["nomad_distance"] for _, _, d in H.edges(data=True)]
            print(f"  Sparse graph edges: {H.number_of_edges()}")
            print(f"  Avg degree: {np.mean(degrees):.1f}")
            print(f"  NoMaD distance range: [{np.min(nomad_dists):.2f}, {np.max(nomad_dists):.2f}]")
        else:
            print(f"  Sparse graph has no edges (threshold too low?)")

        key = f"threshold_{threshold}"
        results[key] = {
            "retained_nodes_indices": retained,
            "num_retained": len(retained),
            "num_map_nodes": len(retained),
            "total_nodes": total_nodes,
            "coverage_ratio": coverage_ratio,
            "distance_threshold": threshold,
        }
        if args.refine:
            results[key]["refined_ratio"] = len(retained) / total_nodes * 100

    # Save results
    output_path = os.path.join(args.data_dir, args.scene, "sparse_graph.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSparse graph results saved to {output_path}")
    for key, val in results.items():
        print(f"  {key}: {val['num_retained']}/{val['total_nodes']} nodes "
              f"({val['coverage_ratio']:.1f}%)")


if __name__ == "__main__":
    main()