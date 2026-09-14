# Bayesian Network Structure Learning for Gene-Regulatory Data

This repository presents a reproducible reimplementation of a Probabilistic Graphical Models course project on **Bayesian-network structure learning** from simulated gene-expression data. The project contrasts two experimental regimes—multifactorial perturbations and gene knockdowns—and evaluates inferred directed acyclic graphs (DAGs) against provided gold-standard networks.

The work demonstrates an end-to-end inference workflow: discretising continuous expression measurements, scoring candidate Bayesian networks, sampling DAG structures with Metropolis–Hastings, estimating posterior edge probabilities, and evaluating recovered edges against a reference graph.

> **Scope.** This is a course project, not a biological claim. The datasets are simulated benchmark inputs; network-recovery results should be interpreted as a methodological exercise.

## Research question

How effectively can a score-based Bayesian-network sampler recover a known gene-regulatory graph from discretised expression data, and how does recovery differ between multifactorial and knockdown perturbation settings?

## Method

The maintained implementation in `pgm_network.py` uses the following pipeline:

1. Independently discretise each gene into two expression states with deterministic K-means clustering.
2. Represent a candidate graph with an adjacency matrix where `A[i, j] = 1` denotes `gene i → gene j`.
3. Score a DAG using a decomposable BIC objective: the conditional maximum-likelihood score minus a penalty for the number of conditional-probability-table parameters.
4. Sample graph structures with a symmetric single-edge-toggle Metropolis–Hastings proposal, rejecting self-loops and cycles.
5. Discard a pre-specified burn-in, compute posterior edge-inclusion probabilities, and threshold them to obtain a consensus graph.
6. Compare off-diagonal edges with the gold-standard graph using precision, recall, F1, and accuracy.

The BIC penalty and the fixed, non-interactive burn-in make the inference target and the run reproducible. They address important limitations of the archived class notebooks, whose unpenalised likelihood favors denser graphs and whose burn-in was selected manually.

## Repository layout

```
pgm_network.py                 Reproducible implementation and command-line runner
tests/test_pgm_network.py      Synthetic-data correctness tests
multifactorial_14004110.ipynb  Archived original analysis: 10-gene multifactorial setting
knockdowns_14004110.ipynb      Archived original analysis: 100-gene knockdown setting
```

## Reproduce an experiment

Use Python 3.10+ and install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The original benchmark data are not included in this repository. Place them in a local directory (they are ignored by Git) and run, for example:

```bash
python pgm_network.py \
  --data data/raw/10_1_multifactorial.tsv \
  --gold data/raw/10_01_gold.txt \
  --iterations 20000 --burn-in 5000 --chains 4 --seed 2026 \
  --output results/multifactorial.json
```

For the knockdown setting, substitute `100_1_knockdowns.tsv` and `100_01_gold.txt`. The command writes the consensus adjacency matrix, edge posterior probabilities, diagnostics, and evaluation metrics to JSON. The default decision threshold is 0.5; this should be tuned or complemented with precision–recall analysis in a fuller study.

## Interpretation and limitations

This project is intentionally transparent about the boundary between coursework and research:

- Directed edges encode statistical dependencies under the assumed Bayesian-network model; they do **not** by themselves establish causal regulation.
- Binary K-means discretisation trades expression magnitude for a simple discrete conditional-probability model. Alternative discretisation schemes or continuous Bayesian networks are natural extensions.
- A single-edge local proposal can mix slowly on larger graphs. Longer, multiple chains and convergence diagnostics are advisable for the 100-gene setting.
- Gold-standard evaluation excludes diagonal entries, because self-regulatory edges are outside this DAG model.

## Original coursework notebooks

The two notebooks are retained as a record of the submitted PGM class work. `pgm_network.py` is the maintained version for portfolio use: it makes graph orientation explicit, fixes random seeds, avoids manual prompts, records every MCMC state, prevents self-loops, and uses a penalised score.

## Suggested portfolio description

> Developed a reproducible Bayesian-network structure-learning pipeline for simulated gene-regulatory data. Implemented deterministic discretisation, BIC-scored DAG inference with Metropolis–Hastings sampling, posterior edge aggregation, and gold-standard network evaluation across multifactorial and knockdown perturbation settings.

