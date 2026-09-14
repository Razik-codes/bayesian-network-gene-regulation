import unittest

import numpy as np
import pandas as pd

from pgm_network import (
    bic_score,
    discretize_binary,
    edge_metrics,
    is_dag,
    metropolis_hastings,
    posterior_consensus,
)


class PGMNetworkTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "parent": [0, 0, 0, 0, 1, 1, 1, 1],
                "child": [0, 0, 0, 0, 1, 1, 1, 1],
                "noise": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def test_cycle_detection(self):
        self.assertTrue(is_dag(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])))
        self.assertFalse(is_dag(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])))

    def test_bic_prefers_informative_parent(self):
        empty = np.zeros((3, 3), dtype=np.int8)
        linked = empty.copy()
        linked[0, 1] = 1
        self.assertGreater(bic_score(self.data, linked), bic_score(self.data, empty))

    def test_sampler_records_every_iteration_and_returns_dags(self):
        samples, scores, diagnostics = metropolis_hastings(self.data, iterations=20, seed=7)
        self.assertEqual(samples.shape, (20, 3, 3))
        self.assertEqual(scores.shape, (20,))
        self.assertTrue(all(is_dag(sample) for sample in samples))
        self.assertGreaterEqual(diagnostics.acceptance_rate, 0.0)
        self.assertLessEqual(diagnostics.acceptance_rate, 1.0)

    def test_consensus_and_metrics_exclude_diagonal(self):
        samples = np.array([[[0, 1], [0, 0]], [[0, 1], [0, 0]]], dtype=np.int8)
        probabilities, consensus = posterior_consensus(samples, burn_in=0)
        self.assertEqual(probabilities[0, 1], 1.0)
        self.assertEqual(consensus[0, 1], 1)
        metrics = edge_metrics(consensus, np.array([[1, 1], [0, 1]], dtype=np.int8))
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_discretization_uses_high_state_for_larger_values(self):
        discrete = discretize_binary(pd.DataFrame({"x": [0.0, 0.1, 10.0, 11.0]}), random_state=1)
        self.assertEqual(discrete["x"].tolist(), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
