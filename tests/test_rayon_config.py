"""Rayon settings must be applied independently before native pool creation."""
import os
import unittest
from unittest.mock import patch

from zkinfer.runtime.rayon_config import configure_coordinator_rayon, rayon_environment


class RayonConfigTests(unittest.TestCase):
    def test_independent_environment_and_auto(self):
        original = {"RAYON_NUM_THREADS": "128", "RAYON_RS_NUM_CPUS": "64", "PATH": "/bin"}
        self.assertEqual(rayon_environment(16, original),
                         {"RAYON_NUM_THREADS": "16", "PATH": "/bin"})
        self.assertEqual(rayon_environment(None, original), {"PATH": "/bin"})
        self.assertEqual(original["RAYON_NUM_THREADS"], "128")

    def test_coordinator_environment_does_not_constrain_prover(self):
        with patch.dict(os.environ, {"RAYON_NUM_THREADS": "128"}, clear=True):
            configure_coordinator_rayon(4)
            self.assertEqual(os.environ["RAYON_NUM_THREADS"], "4")
            self.assertEqual(rayon_environment(16)["RAYON_NUM_THREADS"], "16")
            self.assertNotIn("RAYON_NUM_THREADS", rayon_environment(None))
            self.assertEqual(os.environ["RAYON_NUM_THREADS"], "4")
            configure_coordinator_rayon(None)
            self.assertNotIn("RAYON_NUM_THREADS", os.environ)
