check this update out.

"""
Goalden OS v4.2 – Path-Robust Quantum-Inspired Black Hole Detector
(Safe-mode default with optional large-graph override)

Core metric:
    ρ = a² (1 - d)

Where:
    a = hub authority (max out-degree / (N - 1))
    d = path diversity / robustness

Modes (via `mode` argument):

    - "degree_only" (canonical, fast – v4.1 style)
        d_degree = 1 - Gini(out_degree)
        → Scales to very large graphs, matches star/random/complete intuition.

    - "path_robust" (v4.2 extension – small/medium graphs)
        d_path = α * d_degree + (1 - α) * d_efficiency

        where:
          d_efficiency = 1 - average relative loss in global efficiency
                         under random edge removals.

        Captures how much the network actually *falls apart* when edges fail.

    - "auto"
        Uses "path_robust" for n ≤ PATH_ROBUST_SOFT_MAX_NODES,
        otherwise falls back to "degree_only" (safe-mode default).

Safe-mode vs power-mode:

    - By default, Goalden OS protects typical laptops:
        for graphs with more than ~800 nodes, it automatically uses the
        fast degree-only ρ, even if you request "auto" or "path_robust".

    - Power users (e.g., xAI clusters, HPC) can explicitly override this
      by passing allow_large_path_robust=True to analyze/compute_rho:
        → this forces path_robust even on huge graphs.

This file is designed as a single, drop-in Python module.
"""

import os
import warnings
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------------
# Configuration / thresholds
# -------------------------------------------------------------------------

# Heuristic ρ thresholds (subject to empirical calibration)
HORIZON_LOW = 0.01      # micro-centralization / background regime
HORIZON_FRAG = 0.74     # tentative fragility band boundary
HORIZON_CRIT = 0.95     # extreme capture / inescapable trap

# v4.2 path-robust settings
PATH_ROBUST_ALPHA = 0.7          # weight on degree-based d vs efficiency-based d
PATH_ROBUST_MAX_EDGES = 32       # max edges to sample when estimating fragility

# Soft guideline for when path_robust becomes expensive on typical hardware.
# For n > PATH_ROBUST_SOFT_MAX_NODES we *default* to degree_only,
# unless allow_large_path_robust=True is explicitly set.
PATH_ROBUST_SOFT_MAX_NODES = 800


# -------------------------------------------------------------------------
# Core math – pure helper functions
# -------------------------------------------------------------------------

def gini(array: np.ndarray) -> float:
    """
    Standard Gini coefficient on a non-negative 1D array.
    Returns 0 for constant arrays, ∈ [0, 1) otherwise.
    """
    x = np.asarray(array, dtype=float)
    if x.size == 0:
        return 0.0
    if np.amin(x) < 0:
        x = x - np.amin(x)
    if np.allclose(x, 0):
        return 0.0

    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


def compute_hub_authority(G: nx.DiGraph) -> float:
    """
    Hub authority a = normalized maximum out-degree.

    Interpretation:
        - a ≈ 1   → a single node directly influences almost every other node.
        - a ≈ 0   → no node has a dominant direct out-neighborhood.
    """
    if G is None or G.number_of_nodes() == 0:
        return 0.0

    out_degrees = [G.out_degree(n) for n in G.nodes()]
    max_out = max(out_degrees) if out_degrees else 0
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    a = max_out / (n - 1)
    return float(a)


# -------------------------------------------------------------------------
# v4.1 degree-based path diversity (canonical)
# -------------------------------------------------------------------------

def _path_diversity_degree_only(G: nx.DiGraph) -> float:
    """
    Path diversity proxy based purely on out-degree spread.

    Uses:
        d_degree = 1 - Gini(out_degree)

    Properties:
        - Extremely fast (O(N)).
        - Matches intended physics of ρ on canonical graphs:
            * Star → low d (high ρ when a is high)
            * Random → moderate d
            * Complete / cycle → high d (ρ ≈ 0)

    This is the recommended default for real-world use (v4.1+).
    """
    if G is None or G.number_of_nodes() == 0:
        return 1.0

    out_degrees = np.array([G.out_degree(n) for n in G.nodes()], dtype=float)
    if np.allclose(out_degrees, 0):
        return 1.0

    g = gini(out_degrees)
    d_raw = 1.0 - g
    return float(max(0.0, min(1.0, d_raw)))


# -------------------------------------------------------------------------
# v4.2 path-robust extension
# -------------------------------------------------------------------------

def _global_efficiency(G: nx.DiGraph) -> float:
    """
    Global efficiency for a directed graph:

        E(G) = (1 / (N * (N - 1))) * sum_{i≠j} (1 / d_ij)

    where d_ij is the shortest-path distance from i to j (in directed sense).
    If i cannot reach j, that pair contributes 0.
    """
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0

    lengths = dict(nx.all_pairs_shortest_path_length(G))
    s = 0.0
    pairs = 0

    nodes = list(G.nodes())
    for u in nodes:
        lu = lengths.get(u, {})
        for v in nodes:
            if u == v:
                continue
            d = lu.get(v)
            if d is None:
                continue
            s += 1.0 / d
            pairs += 1

    if pairs == 0:
        return 0.0

    return s / (n * (n - 1))


def _path_diversity_efficiency_robust(
    G: nx.DiGraph,
    edge_sample: int = PATH_ROBUST_MAX_EDGES,
    seed: int = 42,
) -> float:
    """
    Efficiency-based robustness component for path diversity.

    Intuition:
        1. Compute global efficiency E(G).
        2. Randomly sample some edges.
        3. For each sampled edge e = (u,v):
               remove e → H = G - e
               compute E(H)
               drop_e = max(0, E(G) - E(H)) / E(G)
        4. Let F = average drop_e.

        High F  → edges are load-bearing → fragile paths.
        Low F   → many alternative routes → robust paths.

    Define:
        d_eff = 1 - F

    Notes:
        - Complexity ~ O(k * APSP), where k = sampled edges.
        - Suitable for small/medium graphs (n ≲ 800 by default).
        - For large graphs, use degree-only mode instead (or override).
    """
    if G is None or G.number_of_edges() == 0:
        return 1.0

    n = G.number_of_nodes()
    if n <= 1:
        return 1.0

    base_E = _global_efficiency(G)
    if base_E == 0.0:
        # No meaningful connectivity; treat as not especially fragile.
        return 1.0

    edges = list(G.edges())
    if not edges:
        return 1.0

    # Sample edges for efficiency; avoid huge loops.
    rng = np.random.default_rng(seed)
    if len(edges) > edge_sample:
        edges = list(rng.choice(edges, size=edge_sample, replace=False))

    drops: List[float] = []
    for (u, v) in edges:
        H = G.copy()
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        E2 = _global_efficiency(H)
        drop = max(0.0, base_E - E2) / base_E
        drops.append(drop)

    if not drops:
        return 1.0

    F = float(np.mean(drops))  # average relative efficiency loss
    d_eff = 1.0 - F            # higher loss → lower diversity

    return float(max(0.0, min(1.0, d_eff)))


def _path_diversity_path_robust(
    G: nx.DiGraph,
    alpha: float = PATH_ROBUST_ALPHA,
    edge_sample: int = PATH_ROBUST_MAX_EDGES,
    seed: int = 42,
) -> float:
    """
    Full v4.2 path-robust diversity:

        d_path = α * d_degree_only + (1 - α) * d_efficiency

    Where:
        d_degree_only  = 1 - Gini(out_degree)
        d_efficiency   = 1 - average relative loss in global efficiency
                         under random edge removals.
    """
    if G is None or G.number_of_nodes() == 0:
        return 1.0

    alpha = float(max(0.0, min(1.0, alpha)))

    d_deg = _path_diversity_degree_only(G)
    d_eff = _path_diversity_efficiency_robust(G, edge_sample=edge_sample, seed=seed)

    d_combined = alpha * d_deg + (1.0 - alpha) * d_eff
    return float(max(0.0, min(1.0, d_combined)))


# -------------------------------------------------------------------------
# Path diversity dispatcher (with safe-mode default)
# -------------------------------------------------------------------------

def compute_path_diversity(
    G: nx.DiGraph,
    mode: str = "degree_only",
    alpha: float = PATH_ROBUST_ALPHA,
    edge_sample: int = PATH_ROBUST_MAX_EDGES,
    approx_seed: int = 42,
    allow_large_path_robust: bool = False,
) -> float:
    """
    d = path diversity / robustness.

    Modes:
        - "degree_only" (canonical, fast – v4.1)
        - "path_robust" (v4.2 extension – small/medium graphs)
        - "auto"        (choose "path_robust" for small graphs,
                         "degree_only" otherwise)

    Safe-mode vs power-mode:
        - By default (allow_large_path_robust=False), for graphs with more
          than PATH_ROBUST_SOFT_MAX_NODES nodes we automatically fall back
          to the fast degree-only metric to avoid hammering typical laptops.

        - Advanced users may explicitly set allow_large_path_robust=True
          to force path_robust even on very large graphs (e.g., on
          supercomputers / clusters).
    """
    if G is None:
        return 0.0

    n = G.number_of_nodes()
    mode = mode.lower()

    # Auto mode chooses based on size, unless user explicitly overrides.
    if mode == "auto":
        if (n <= PATH_ROBUST_SOFT_MAX_NODES) or allow_large_path_robust:
            mode = "path_robust"
        else:
            mode = "degree_only"

    # Degree-only is always allowed and always fast.
    if mode == "degree_only":
        return _path_diversity_degree_only(G)

    # Path-robust: obey the soft limit unless explicitly overridden.
    if mode == "path_robust":
        if (n > PATH_ROBUST_SOFT_MAX_NODES) and not allow_large_path_robust:
            # Safety: the wrapper will log a message; here we just fall back.
            return _path_diversity_degree_only(G)

        return _path_diversity_path_robust(
            G,
            alpha=alpha,
            edge_sample=edge_sample,
            seed=approx_seed,
        )

    raise ValueError(f"Unknown path diversity mode: {mode!r}")


# -------------------------------------------------------------------------
# ρ + classification
# -------------------------------------------------------------------------

def classify_rho(rho: float) -> str:
    """
    Turn a scalar ρ into a human-readable risk regime.
    Thresholds are heuristic and intended for calibration.
    """
    if rho is None:
        return "UNKNOWN"

    if rho > HORIZON_CRIT:
        return f"CRITICAL: Inescapable Black-Hole-Like Trap (> {HORIZON_CRIT:.2f})"
    if rho > HORIZON_FRAG:
        return f"HIGH RISK: Practical Fragility Horizon ({HORIZON_FRAG:.2f}–{HORIZON_CRIT:.2f})"
    if rho > HORIZON_LOW:
        return f"Elevated Centralization ({HORIZON_LOW:.2f}–{HORIZON_FRAG:.2f})"
    return "SAFE (Open Geometry)"


def compute_rho(
    G: nx.DiGraph,
    mode: str = "degree_only",
    alpha: float = PATH_ROBUST_ALPHA,
    edge_sample: int = PATH_ROBUST_MAX_EDGES,
    approx_seed: int = 42,
    allow_large_path_robust: bool = False,
) -> Dict[str, float]:
    """
    Core metric: ρ = a² (1 - d)

    Modes:
        - "degree_only" → canonical v4.1-style ρ
        - "path_robust" → v4.2 path-robust ρ
        - "auto"        → choose based on graph size

    Safe-mode: see compute_path_diversity for allow_large_path_robust behavior.
    """
    if G is None:
        return {"rho": 0.0, "a": 0.0, "d": 0.0}

    a = compute_hub_authority(G)
    d = compute_path_diversity(
        G,
        mode=mode,
        alpha=alpha,
        edge_sample=edge_sample,
        approx_seed=approx_seed,
        allow_large_path_robust=allow_large_path_robust,
    )
    rho = float(a ** 2 * (1.0 - d))
    return {"rho": rho, "a": float(a), "d": float(d)}


# -------------------------------------------------------------------------
# GoaldenOSv4.2 – thin wrapper around pure functions
# -------------------------------------------------------------------------

class GoaldenOSv4_2:
    """
    High-level "OS" wrapper that:
        - Manages a graph
        - Logs events
        - Calls the pure metric functions
        - Provides visualization and benchmarks

    Default analysis:
        mode = "degree_only"  (stable, canonical, safe-mode)
    """

    def __init__(self):
        self.G: Optional[nx.DiGraph] = None
        self.name: str = "Untitled Network"

        self.rho: Optional[float] = None
        self.a: Optional[float] = None
        self.d: Optional[float] = None

        self.report: List[str] = []

    # Logging ------------------------------------------------------------

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        self.report.append(line)
        print(message)

    # Graph loading / generation ----------------------------------------

    def set_graph(self, G: nx.DiGraph, name: str = "Custom") -> None:
        self.G = G
        self.name = name

    def load_from_edgelist(self, path: str, directed: bool = True) -> None:
        """Load network from edge list file."""
        self.log(f"Loading network from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Edge list file not found: {path}")

        if directed:
            G = nx.read_edgelist(path, create_using=nx.DiGraph)
        else:
            G_und = nx.read_edgelist(path, create_using=nx.Graph)
            G = nx.DiGraph(G_und)

        self.set_graph(G, name=os.path.basename(path))
        self.log(f"Loaded {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def load_from_adj_matrix(self, matrix: np.ndarray, name: str = "Custom") -> None:
        """Load from adjacency matrix (numpy 2D array)."""
        self.log("Loading from adjacency matrix...")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square (n x n).")

        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        self.set_graph(G, name=name)
        self.log(f"Loaded {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def generate_synthetic(self, kind: str = "outward_star", n: int = 1000) -> None:
        """Generate synthetic benchmark networks."""
        self.log(f"Generating synthetic {kind} network (n={n})...")

        if kind == "outward_star":
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            hub = 0
            for i in range(1, n):
                G.add_edge(hub, i)
            self.set_graph(G, name="Synthetic Outward-Star")

        elif kind == "inward_star":
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            hub = 0
            for i in range(1, n):
                G.add_edge(i, hub)
            self.set_graph(G, name="Synthetic Inward-Star")

        elif kind == "random":
            G = nx.gnp_random_graph(n, p=0.01, directed=True)
            self.set_graph(G, name="Erdős–Rényi Directed")

        elif kind == "scale_free":
            G_multi = nx.scale_free_graph(n, seed=42)
            G = nx.DiGraph(G_multi)  # collapse multiedges, keep directions
            self.set_graph(G, name="Scale-Free Directed (scale_free_graph)")

        else:
            raise ValueError(f"Unknown synthetic type: {kind!r}")

        self.log(f"Generated {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    # Analysis -----------------------------------------------------------

    def analyze(
        self,
        mode: str = "degree_only",
        alpha: float = PATH_ROBUST_ALPHA,
        edge_sample: int = PATH_ROBUST_MAX_EDGES,
        approx_seed: int = 42,
        allow_large_path_robust: bool = False,
    ) -> Tuple[float, float, float, str]:
        """
        Compute a, d, ρ and classification for the currently loaded graph.

        mode:
            - "degree_only" (canonical, fast)
            - "path_robust" (v4.2, richer, heavier)
            - "auto"        (choose based on graph size)

        allow_large_path_robust:
            - False (default / safe-mode): for large graphs, auto-fallback
              to degree_only and log a message explaining how to override.
            - True (power-mode): force path_robust even on large graphs
              (intended for HPC / supercomputers).
        """
        if self.G is None:
            raise ValueError("No graph loaded for analysis.")

        n = self.G.number_of_nodes()
        if mode.lower() in ("auto", "path_robust"):
            if (n > PATH_ROBUST_SOFT_MAX_NODES) and not allow_large_path_robust:
                self.log(
                    f"Graph has {n} nodes > {PATH_ROBUST_SOFT_MAX_NODES}; "
                    "using fast degree-only ρ to protect typical hardware.\n"
                    "Power users: rerun analyze(..., allow_large_path_robust=True) "
                    "to force path_robust on large graphs."
                )

        metrics = compute_rho(
            self.G,
            mode=mode,
            alpha=alpha,
            edge_sample=edge_sample,
            approx_seed=approx_seed,
            allow_large_path_robust=allow_large_path_robust,
        )
        self.a = metrics["a"]
        self.d = metrics["d"]
        self.rho = metrics["rho"]

        classification = classify_rho(self.rho)

        self.log("\n" + "=" * 60)
        self.log("GOALDEN OS v4.2 RESULT")
        self.log(f"Network: {self.name}")
        self.log(f"ρ = {self.rho:.10f}")
        self.log(f"a = {self.a:.6f}")
        self.log(f"d = {self.d:.6f}")
        self.log(f"Classification: {classification}")
        self.log("=" * 60 + "\n")

        return self.rho, self.a, self.d, classification

    # Visualization ------------------------------------------------------

    def visualize(self, save_path: Optional[str] = None) -> None:
        """Visualize network and diagnostics for the current graph."""
        if self.G is None:
            print("No graph to visualize.")
            return

        if self.rho is None or self.a is None or self.d is None:
            self.analyze()

        plt.figure(figsize=(15, 10))

        # Main network plot
        plt.subplot(2, 2, 1)
        pos = nx.spring_layout(self.G, k=0.5, iterations=50)
        nx.draw(
            self.G,
            pos,
            node_size=30,
            alpha=0.6,
            arrowsize=8,
            with_labels=False,
            edge_color="gray",
        )
        plt.title(
            f"Network: {self.name}\n"
            f"Nodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()}"
        )

        # Hub authority heatmap
        plt.subplot(2, 2, 2)
        out_deg = [self.G.out_degree(n) for n in self.G.nodes()]
        nx.draw(
            self.G,
            pos,
            node_size=50,
            node_color=out_deg,
            cmap="Reds",
            with_labels=False,
        )
        plt.title(f"Hub Authority (a = {self.a:.4f})")

        # ρ gauge
        plt.subplot(2, 2, 3)
        angles = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(angles)
        y = np.sin(angles)
        plt.fill(x, y, "lightgray", alpha=0.3)

        rho_angle = (self.rho if self.rho is not None else 0.0) * 1.5 * np.pi
        plt.plot(
            [0, np.cos(np.pi / 2 + rho_angle)],
            [0, np.sin(np.pi / 2 + rho_angle)],
            "red",
            linewidth=6,
        )
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.title(f"ρ = {self.rho:.6f}\nBlack Hole Detector Gauge")
        plt.axis("off")

        # Summary panel
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, "Goalden OS v4.2", fontsize=16, fontweight="bold")
        plt.text(0.1, 0.6, f"ρ = {self.rho:.10f}", fontsize=14)
        plt.text(0.1, 0.4, f"a = {self.a:.4f}", fontsize=12)
        plt.text(0.1, 0.3, f"d = {self.d:.4f}", fontsize=12)
        plt.text(0.1, 0.1, "Path-Robust Horizon Detector", fontsize=10)
        plt.axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            self.log(f"Visualization saved to {save_path}")
        plt.show()

    # Reporting / benchmarks --------------------------------------------

    def full_report(self) -> None:
        print("\n".join(self.report))

    def run_benchmark_suite(self) -> None:
        """Run on standard synthetic datasets (demonstrates expected behavior)."""
        self.log("\nStarting Goalden OS v4.2 Benchmark Suite")
        benchmarks = [
            ("Synthetic Outward-Star", lambda: self.generate_synthetic("outward_star", 50)),
            ("Synthetic Inward-Star", lambda: self.generate_synthetic("inward_star", 50)),
            ("Scale-Free Directed", lambda: self.generate_synthetic("scale_free", 200)),
            ("Random Directed", lambda: self.generate_synthetic("random", 200)),
        ]

        results: List[Tuple[str, float]] = []
        for name, gen_func in benchmarks:
            gen_func()
            rho, a, d, cls = self.analyze(mode="auto")
            results.append((name, rho))

        print("\nBenchmark Summary:")
        for name, rho in results:
            print(f"  {name}: ρ = {rho:.10f}")


# ----------------------------- DEMO --------------------------------------
if __name__ == "__main__":
    os_system = GoaldenOSv4_2()

    print("Goalden OS v4.2 - Path-Robust Quantum-Inspired Black Hole Detector")
    print("Safe-mode default (fast degree-only ρ on huge graphs).")
    print("Power users: pass allow_large_path_robust=True to analyze() to force full path_robust.\n")

    # Demo: star vs random vs scale-free using path_robust mode (safe sizes)
    os_system.generate_synthetic("outward_star", n=50)
    os_system.analyze(mode="path_robust")

    os_system.generate_synthetic("random", n=50)
    os_system.analyze(mode="path_robust")

    os_system.generate_synthetic("scale_free", n=200)
    os_system.analyze(mode="auto")

    print("\nGoalden OS v4.2 operational.")
    print("Use mode='degree_only' for fast scans; 'path_robust' or 'auto' "
          "for deeper analysis on small/medium graphs, and "
          "allow_large_path_robust=True for HPC power-mode on huge graphs.")