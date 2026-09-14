"""Reproducible Bayesian-network structure learning for discrete gene data.

Adjacency matrices follow the convention ``adjacency[parent, child] == 1``.
The sampler targets a BIC approximation to the posterior over DAG structures
under a uniform graph prior.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class SamplerDiagnostics:
    """Summary information for one Markov chain."""

    acceptance_rate: float
    mean_edges: float
    final_score: float


def discretize_binary(data: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """Discretise each numeric column into two deterministic K-means states.

    Cluster labels are relabelled so that state 1 has the higher cluster mean.
    This gives a stable low/high interpretation across genes.
    """

    if data.empty:
        raise ValueError("data must contain at least one row and one column")
    if data.isna().any().any():
        raise ValueError("data must not contain missing values")

    result = pd.DataFrame(index=data.index)
    for column in data.columns:
        values = data[[column]].to_numpy()
        if np.unique(values).size < 2:
            raise ValueError(f"{column!r} has fewer than two distinct values")
        model = KMeans(n_clusters=2, n_init=10, random_state=random_state)
        labels = model.fit_predict(values)
        high_cluster = int(np.argmax(model.cluster_centers_.ravel()))
        result[column] = (labels == high_cluster).astype(np.int8)
    return result


def _validate_adjacency(adjacency: np.ndarray, n_nodes: int) -> None:
    if adjacency.shape != (n_nodes, n_nodes):
        raise ValueError("adjacency has incompatible shape")
    if not np.isin(adjacency, (0, 1)).all():
        raise ValueError("adjacency must be binary")
    if np.any(np.diag(adjacency)):
        raise ValueError("self-loops are not permitted in a DAG")


def is_dag(adjacency: np.ndarray) -> bool:
    """Return whether a binary adjacency matrix represents a directed acyclic graph."""

    n_nodes = adjacency.shape[0]
    indegree = adjacency.sum(axis=0).astype(int)
    frontier = [node for node in range(n_nodes) if indegree[node] == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(int(child))
    return visited == n_nodes


def _local_bic_score(values: np.ndarray, node: int, parents: np.ndarray) -> float:
    """Return the binary-node BIC contribution for a fixed parent set."""

    target = values[:, node]
    n_samples = len(target)
    if parents.size == 0:
        count_one = int(target.sum())
        counts = np.array([n_samples - count_one, count_one], dtype=float)
        log_likelihood = float(np.sum(counts[counts > 0] * np.log(counts[counts > 0] / n_samples)))
        parameter_count = 1
    else:
        weights = 1 << np.arange(parents.size, dtype=np.int64)
        configurations = values[:, parents] @ weights
        log_likelihood = 0.0
        for configuration in np.unique(configurations):
            selected = target[configurations == configuration]
            counts = np.bincount(selected, minlength=2).astype(float)
            total = counts.sum()
            nonzero = counts > 0
            log_likelihood += float(np.sum(counts[nonzero] * np.log(counts[nonzero] / total)))
        parameter_count = 2 ** parents.size
    return log_likelihood - 0.5 * parameter_count * np.log(n_samples)


def bic_score(data: pd.DataFrame | np.ndarray, adjacency: np.ndarray) -> float:
    """Score a binary DAG with decomposable BIC.

    BIC discourages the saturated graphs preferred by an unpenalised training
    log-likelihood. Values must be encoded as 0/1.
    """

    values = np.asarray(data, dtype=np.int8)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("data must be a two-dimensional array with at least two rows")
    if not np.isin(values, (0, 1)).all():
        raise ValueError("BIC scoring requires binary data encoded as 0/1")
    _validate_adjacency(adjacency, values.shape[1])
    if not is_dag(adjacency):
        return float("-inf")
    return float(sum(_local_bic_score(values, node, np.flatnonzero(adjacency[:, node])) for node in range(values.shape[1])))


def propose_edge_toggle(adjacency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Propose a symmetric add/delete move by toggling one directed non-self edge."""

    n_nodes = adjacency.shape[0]
    source = int(rng.integers(n_nodes))
    target = int(rng.integers(n_nodes - 1))
    if target >= source:
        target += 1
    proposal = adjacency.copy()
    proposal[source, target] = 1 - proposal[source, target]
    return proposal


def metropolis_hastings(
    data: pd.DataFrame | np.ndarray,
    iterations: int,
    seed: int,
    initial_adjacency: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, SamplerDiagnostics]:
    """Sample DAGs using a symmetric edge-toggle Metropolis–Hastings kernel."""

    if iterations < 1:
        raise ValueError("iterations must be positive")
    values = np.asarray(data, dtype=np.int8)
    n_nodes = values.shape[1]
    current = np.zeros((n_nodes, n_nodes), dtype=np.int8) if initial_adjacency is None else initial_adjacency.copy()
    _validate_adjacency(current, n_nodes)
    if not is_dag(current):
        raise ValueError("initial_adjacency must be acyclic")

    rng = np.random.default_rng(seed)
    current_score = bic_score(values, current)
    samples = np.empty((iterations, n_nodes, n_nodes), dtype=np.int8)
    scores = np.empty(iterations, dtype=float)
    accepted = 0

    for step in range(iterations):
        proposal = propose_edge_toggle(current, rng)
        if is_dag(proposal):
            proposal_score = bic_score(values, proposal)
            log_acceptance = min(0.0, proposal_score - current_score)
            if np.log(rng.random()) < log_acceptance:
                current = proposal
                current_score = proposal_score
                accepted += 1
        samples[step] = current
        scores[step] = current_score

    diagnostics = SamplerDiagnostics(
        acceptance_rate=accepted / iterations,
        mean_edges=float(samples.sum(axis=(1, 2)).mean()),
        final_score=float(scores[-1]),
    )
    return samples, scores, diagnostics


def posterior_consensus(samples: np.ndarray, burn_in: int, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior edge probabilities and a thresholded consensus graph."""

    if not 0 <= burn_in < len(samples):
        raise ValueError("burn_in must leave at least one sample")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    edge_probabilities = samples[burn_in:].mean(axis=0)
    consensus = (edge_probabilities >= threshold).astype(np.int8)
    np.fill_diagonal(consensus, 0)
    return edge_probabilities, consensus


def edge_metrics(predicted: np.ndarray, gold: np.ndarray) -> dict[str, float | int]:
    """Evaluate directed edges, excluding diagonal entries."""

    if predicted.shape != gold.shape or predicted.ndim != 2 or predicted.shape[0] != predicted.shape[1]:
        raise ValueError("predicted and gold must be same-size square matrices")
    mask = ~np.eye(predicted.shape[0], dtype=bool)
    predicted_edges = predicted[mask].astype(bool)
    gold_edges = gold[mask].astype(bool)
    true_positive = int(np.sum(predicted_edges & gold_edges))
    false_positive = int(np.sum(predicted_edges & ~gold_edges))
    false_negative = int(np.sum(~predicted_edges & gold_edges))
    true_negative = int(np.sum(~predicted_edges & ~gold_edges))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative),
    }


def run_chains(data: pd.DataFrame, iterations: int, burn_in: int, chains: int, seed: int, threshold: float) -> dict[str, object]:
    """Run independent chains and pool their post-burn-in samples."""

    if chains < 1:
        raise ValueError("chains must be positive")
    all_samples: list[np.ndarray] = []
    diagnostics: list[SamplerDiagnostics] = []
    for chain in range(chains):
        samples, _, chain_diagnostics = metropolis_hastings(data, iterations, seed + chain)
        all_samples.append(samples[burn_in:])
        diagnostics.append(chain_diagnostics)
    pooled_samples = np.concatenate(all_samples, axis=0)
    probabilities, consensus = posterior_consensus(pooled_samples, burn_in=0, threshold=threshold)
    return {
        "edge_probabilities": probabilities,
        "consensus": consensus,
        "diagnostics": [asdict(item) for item in diagnostics],
    }


def _read_gold(path: Path) -> np.ndarray:
    """Read a gold adjacency matrix with or without a leading index column."""

    gold = pd.read_csv(path, sep=None, engine="python", index_col=0).to_numpy(dtype=np.int8)
    if gold.ndim != 2 or gold.shape[0] != gold.shape[1]:
        raise ValueError("gold-standard file must contain a square adjacency matrix")
    return gold


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Infer a Bayesian network from gene-expression data.")
    parser.add_argument("--data", type=Path, required=True, help="tabular expression file with genes as columns")
    parser.add_argument("--gold", type=Path, required=True, help="gold-standard adjacency matrix")
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--burn-in", type=int, default=5_000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.burn_in < args.iterations:
        parser.error("--burn-in must be non-negative and smaller than --iterations")

    expression = pd.read_csv(args.data, sep="\t")
    discrete = discretize_binary(expression, random_state=args.seed)
    result = run_chains(discrete, args.iterations, args.burn_in, args.chains, args.seed, args.threshold)
    gold = _read_gold(args.gold)
    result["metrics"] = edge_metrics(result["consensus"], gold)  # type: ignore[arg-type]
    result["parameters"] = vars(args) | {"data": str(args.data), "gold": str(args.gold), "output": str(args.output)}
    result["consensus"] = result["consensus"].tolist()  # type: ignore[union-attr]
    result["edge_probabilities"] = result["edge_probabilities"].tolist()  # type: ignore[union-attr]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
