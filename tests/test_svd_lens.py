"""
Tests for SVD-Lens module.

Comprehensive test suite covering SVD analysis,
bottleneck detection, and runtime monitoring.
"""

import pytest
import torch
import torch.nn as nn
from svd_lens import (
    SVDLens,
    SVDLensHook,
    SVDEfficientLens,
    SVDResult,
    BottleneckAnalysis,
    LayerLens,
    create_svd_lens,
)


class TestSVDResult:
    """Test SVDResult dataclass."""

    def test_svd_result_creation(self):
        """Test creating SVDResult."""
        result = SVDResult(
            U=torch.randn(100, 10),
            S=torch.randn(10),
            Vt=torch.randn(10, 100),
            explained_variance=torch.randn(10),
            cumulative_variance=torch.randn(10)
        )
        
        assert result.U.shape == (100, 10)
        assert result.S.shape == (10,)
        assert result.Vt.shape == (10, 100)

    def test_svd_result_str(self):
        """Test SVDResult string representation."""
        result = SVDResult(
            U=torch.randn(100, 10),
            S=torch.randn(10),
            Vt=torch.randn(10, 100),
            explained_variance=torch.tensor([0.1, 0.2, 0.7]),
            cumulative_variance=torch.tensor([0.1, 0.3, 1.0])
        )
        
        str_repr = str(result)
        assert "SVDResult" in str_repr


class TestBottleneckAnalysis:
    """Test BottleneckAnalysis dataclass."""

    def test_bottleneck_analysis_creation(self):
        """Test creating BottleneckAnalysis."""
        analysis = BottleneckAnalysis(
            layer_idx=0,
            layer_name="test_layer",
            input_dim=100,
            output_dim=200,
            rank=50,
            effective_rank=45.5,
            compression_ratio=4.0,
            top_k_variance={1: 30.0, 5: 60.0, 10: 80.0},
            is_bottleneck=True,
            bottleneck_strength=0.85
        )
        
        assert analysis.layer_idx == 0
        assert analysis.rank == 50
        assert analysis.compression_ratio == 4.0
        assert analysis.is_bottleneck is True

    def test_bottleneck_analysis_str(self):
        """Test BottleneckAnalysis string representation."""
        analysis = BottleneckAnalysis(
            layer_idx=0,
            layer_name="test_layer",
            input_dim=100,
            output_dim=200,
            rank=50,
            effective_rank=45.5,
            compression_ratio=4.0,
            is_bottleneck=True
        )
        
        str_repr = str(analysis)
        assert "BottleneckAnalysis" in str_repr


class TestLayerLens:
    """Test LayerLens dataclass."""

    def test_layer_lens_creation(self):
        """Test creating LayerLens."""
        svd_result = SVDResult(
            U=torch.randn(100, 10),
            S=torch.randn(10),
            Vt=torch.randn(10, 100),
            explained_variance=torch.randn(10),
            cumulative_variance=torch.randn(10)
        )
        
        analysis = BottleneckAnalysis(
            layer_idx=0,
            layer_name="test_layer",
            input_dim=100,
            output_dim=200,
            rank=50,
            effective_rank=42.0,
            compression_ratio=2.0
        )
        
        lens = LayerLens(
            layer_idx=0,
            layer_name="test_layer",
            svd_result=svd_result,
            bottleneck_analysis=analysis,
            activation_stats={'mean': 0.0, 'std': 1.0}
        )
        
        assert lens.layer_idx == 0
        assert lens.svd_result is svd_result
        assert lens.bottleneck_analysis is analysis


class TestSVDLens:
    """Test SVDLens class."""

    def test_lens_init(self):
        """Test SVDLens initialization."""
        lens = SVDLens(num_samples=10, threshold=0.95)
        
        assert lens.num_samples == 10
        assert lens.threshold == 0.95
        assert lens.device == 'cpu'

    def test_analyze_simple_layer(self):
        """Test analyzing a simple layer."""
        lens = SVDLens(num_samples=5)
        
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        inputs = {'input': torch.randn(10, 64)}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)
        assert layer_lens.layer_idx == 0
        assert layer_lens.svd_result is not None
        assert layer_lens.bottleneck_analysis is not None

    def test_analyze_model(self):
        """Test analyzing all layers in a model."""
        lens = SVDLens(num_samples=5)
        
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        inputs = {'input': torch.randn(10, 64)}
        lenses = lens.analyze_model(model, inputs=inputs)
        
        assert len(lenses) >= 2

    def test_find_bottlenecks(self):
        """Test finding bottlenecks."""
        lens = SVDLens(num_samples=5, threshold=0.9)
        
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        inputs = {'input': torch.randn(10, 64)}
        bottlenecks = lens.find_bottlenecks(model, inputs=inputs)
        
        assert isinstance(bottlenecks, list)
        for b in bottlenecks:
            assert isinstance(b, BottleneckAnalysis)

    def test_get_compression_potential(self):
        """Test getting compression potential."""
        lens = SVDLens(num_samples=5, threshold=0.95)
        
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        inputs = {'input': torch.randn(10, 64)}
        potential = lens.get_compression_potential(model, inputs=inputs)
        
        assert isinstance(potential, dict)
        assert len(potential) > 0

    def test_get_lens(self):
        """Test getting lens for specific layer."""
        lens = SVDLens(num_samples=5)
        
        model = nn.Sequential(nn.Linear(64, 128))
        
        inputs = {'input': torch.randn(10, 64)}
        lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        layer_lens = lens.get_lens(0)
        
        assert isinstance(layer_lens, LayerLens)

    def test_clear(self):
        """Test clearing stored analyses."""
        lens = SVDLens(num_samples=5)
        
        model = nn.Sequential(nn.Linear(64, 128))
        
        inputs = {'input': torch.randn(10, 64)}
        lens.analyze_model(model, inputs=inputs)
        
        lens.clear()
        
        assert len(lens.get_all_lenses()) == 0


class TestSVDLensHook:
    """Test SVDLensHook class."""

    def test_hook_init(self):
        """Test hook initialization."""
        lens = SVDLens(num_samples=5)
        hook = SVDLensHook(lens, layer_idx=0, window_size=100)
        
        assert hook.lens is lens
        assert hook.layer_idx == 0
        assert hook.window_size == 100

    def test_hook_attach_detach(self):
        """Test attaching and detaching hook."""
        lens = SVDLens(num_samples=5)
        hook = SVDLensHook(lens, layer_idx=0)
        
        layer = nn.Linear(64, 128)
        
        hook.attach(layer)
        
        x = torch.randn(10, 64)
        output = layer(x)
        
        hook.detach()

    def test_hook_context_manager(self):
        """Test hook as context manager."""
        lens = SVDLens(num_samples=5)
        hook = SVDLensHook(lens, layer_idx=0)
        
        layer = nn.Linear(64, 128)
        
        with hook:
            x = torch.randn(10, 64)
            output = layer(x)


class TestSVDEfficientLens:
    """Test SVDEfficientLens class."""

    def test_efficient_lens_init(self):
        """Test efficient lens initialization."""
        lens = SVDEfficientLens(num_samples=5, approximate=True)
        
        assert lens.approximate is True

    def test_efficient_lens_analyze(self):
        """Test efficient lens analysis."""
        lens = SVDEfficientLens(num_samples=5)
        
        model = nn.Sequential(nn.Linear(64, 128))
        
        inputs = {'input': torch.randn(10, 64)}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)


class TestFactoryFunction:
    """Test factory function."""

    def test_create_lens(self):
        """Test creating lens with factory."""
        lens = create_svd_lens(num_samples=5, threshold=0.95)
        
        assert isinstance(lens, SVDLens)
        assert lens.num_samples == 5
        assert lens.threshold == 0.95


class TestEdgeCases:
    """Test edge cases."""

    def test_single_sample(self):
        """Test with single sample."""
        lens = SVDLens(num_samples=1)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(1, 64)}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)

    def test_small_dimensions(self):
        """Test with small dimensions."""
        lens = SVDLens(num_samples=5)
        
        model = nn.Sequential(nn.Linear(16, 32))
        inputs = {'input': torch.randn(5, 16)}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)

    def test_large_dimensions(self):
        """Test with large dimensions."""
        lens = SVDLens(num_samples=3)
        
        model = nn.Sequential(nn.Linear(512, 1024))
        inputs = {'input': torch.randn(3, 512)}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)


class TestNumericalStability:
    """Test numerical stability."""

    def test_very_small_values(self):
        """Test with very small activation values."""
        lens = SVDLens(num_samples=3)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(5, 64) * 1e-10}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)
        assert layer_lens.svd_result is not None

    def test_very_large_values(self):
        """Test with very large activation values."""
        lens = SVDLens(num_samples=3)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.randn(5, 64) * 1e10}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)

    def test_zeros(self):
        """Test with all zeros."""
        lens = SVDLens(num_samples=3)
        
        model = nn.Sequential(nn.Linear(64, 128))
        inputs = {'input': torch.zeros(5, 64)}
        layer_lens = lens.analyze_layer(model, layer_idx=0, inputs=inputs)
        
        assert isinstance(layer_lens, LayerLens)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
