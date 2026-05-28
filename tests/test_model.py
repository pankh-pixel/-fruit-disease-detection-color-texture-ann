"""
tests/test_model.py
===================
Unit tests for the ANN model builder.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_ann, compile_model


class TestBuildANN:
    def test_output_shape(self):
        model = build_ann(input_dim=200, num_classes=5)
        assert model.output_shape == (None, 5)

    def test_parameter_count_positive(self):
        model = build_ann(input_dim=200, num_classes=5)
        assert model.count_params() > 0

    def test_forward_pass(self):
        model = build_ann(input_dim=200, num_classes=5)
        compile_model(model)
        x = np.random.rand(4, 200).astype(np.float32)
        y = model.predict(x, verbose=0)
        assert y.shape == (4, 5)
        # Softmax outputs should sum to ~1 per sample
        np.testing.assert_allclose(y.sum(axis=1), np.ones(4), atol=1e-5)

    def test_invalid_optimizer_raises(self):
        model = build_ann(input_dim=50, num_classes=3)
        with pytest.raises(ValueError):
            compile_model(model, optimizer="invalid_opt")

    def test_custom_hidden_layers(self):
        model = build_ann(input_dim=100, num_classes=4, hidden_layers=[64, 32])
        # Input + 2 Dense-BN-Relu-Dropout blocks + output
        compile_model(model)
        x = np.random.rand(2, 100).astype(np.float32)
        out = model.predict(x, verbose=0)
        assert out.shape == (2, 4)
