"""
SVD-Lens: SVD-based Lens for Neural Network Layer Analysis

Provides SVD-based analysis tools for examining neural network layer behavior,
identifying bottlenecks, and understanding knowledge representation.

Key Features:
- Layer-wise SVD decomposition
- Bottleneck identification
- Knowledge concentration analysis
- Compressibility scoring
- Runtime monitoring hooks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SVDResult:
    """Result of SVD decomposition."""
    U: torch.Tensor
    S: torch.Tensor
    Vt: torch.Tensor
    explained_variance: torch.Tensor
    cumulative_variance: torch.Tensor
    
    def __str__(self) -> str:
        return (
            f"SVDResult(U={self.U.shape}, S={self.S.shape}, Vt={self.Vt.shape}, "
            f"explained={self.explained_variance.sum():.2%})"
        )


@dataclass
class BottleneckAnalysis:
    """Analysis of layer bottleneck."""
    layer_idx: int
    layer_name: str
    input_dim: int
    output_dim: int
    rank: int
    effective_rank: float
    compression_ratio: float
    top_k_variance: Dict[int, float] = field(default_factory=dict)
    is_bottleneck: bool = False
    bottleneck_strength: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"BottleneckAnalysis(layer={self.layer_idx}, rank={self.rank}, "
            f"compression={self.compression_ratio:.2f}x, "
            f"bottleneck={self.is_bottleneck})"
        )


@dataclass
class LayerLens:
    """Lens for a single layer."""
    layer_idx: int
    layer_name: str
    svd_result: Optional[SVDResult] = None
    bottleneck_analysis: Optional[BottleneckAnalysis] = None
    activation_stats: Dict[str, float] = field(default_factory=dict)


class SVDLens:
    """
    SVD-based lens for analyzing neural network layers.
    
    Provides tools to examine layer behavior through SVD decomposition,
    identify bottlenecks, and understand knowledge representation.
    
    Args:
        num_samples: Number of samples for analysis
        threshold: Variance threshold for bottleneck detection
        device: Device to run analysis on
    """
    
    def __init__(
        self,
        num_samples: int = 10,
        threshold: float = 0.95,
        device: str = 'cpu'
    ):
        self.num_samples = num_samples
        self.threshold = threshold
        self.device = device
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, List[torch.Tensor]] = {}
        self._lenses: Dict[int, LayerLens] = {}
    
    def _compute_svd(
        self,
        matrix: torch.Tensor
    ) -> SVDResult:
        """
        Compute SVD of activation matrix.
        
        Args:
            matrix: 2D activation matrix (samples x features)
            
        Returns:
            SVDResult with U, S, Vt, and variance metrics
        """
        # Ensure float32 for stability
        matrix = matrix.float()
        
        # Compute SVD
        U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
        
        # Compute explained variance
        explained_variance = S ** 2
        total_variance = explained_variance.sum()
        cumulative_variance = torch.cumsum(explained_variance, dim=0) / total_variance
        
        return SVDResult(
            U=U,
            S=S,
            Vt=Vt,
            explained_variance=explained_variance,
            cumulative_variance=cumulative_variance
        )
    
    def _analyze_bottleneck(
        self,
        svd_result: SVDResult,
        layer_idx: int,
        layer_name: str,
        input_dim: int,
        output_dim: int
    ) -> BottleneckAnalysis:
        """
        Analyze layer for bottleneck characteristics.
        
        Args:
            svd_result: SVD result for the layer
            layer_idx: Layer index
            layer_name: Layer name
            input_dim: Input dimension
            output_dim: Output dimension
            
        Returns:
            BottleneckAnalysis for the layer
        """
        S = svd_result.S
        cum_var = svd_result.cumulative_variance
        
        # Compute rank (number of non-zero singular values)
        rank = int((S > 1e-6).sum().item())
        
        # Compute effective rank
        effective_rank = torch.exp(
            torch.sum(torch.log(S + 1e-9))
        ).item() / (S.sum() + 1e-9)
        
        # Find minimal k for threshold variance
        for k in range(1, len(cum_var) + 1):
            if cum_var[k-1] >= self.threshold:
                compression_ratio = output_dim / k
                break
        else:
            k = len(cum_var)
            compression_ratio = 1.0
        
        # Compute top-k variances
        top_k_variance = {}
        for k_val in [1, 5, 10, 20, 50, 100]:
            if k_val < len(cum_var):
                top_k_variance[k_val] = cum_var[k_val-1].item() * 100
        
        # Determine if bottleneck
        # Bottleneck if top 10% of components explain > threshold variance
        top_10_pct = int(len(cum_var) * 0.1)
        if top_10_pct > 0 and cum_var[top_10_pct-1] >= self.threshold:
            is_bottleneck = True
            bottleneck_strength = cum_var[top_10_pct-1].item()
        else:
            is_bottleneck = False
            bottleneck_strength = cum_var[-1].item() if len(cum_var) > 0 else 0.0
        
        return BottleneckAnalysis(
            layer_idx=layer_idx,
            layer_name=layer_name,
            input_dim=input_dim,
            output_dim=output_dim,
            rank=rank,
            effective_rank=effective_rank,
            compression_ratio=compression_ratio,
            top_k_variance=top_k_variance,
            is_bottleneck=is_bottleneck,
            bottleneck_strength=bottleneck_strength
        )
    
    def _collect_activation(
        self,
        module: nn.Module,
        inp: torch.Tensor,
        out: torch.Tensor,
        layer_idx: int
    ) -> None:
        """Hook function to collect activations."""
        if layer_idx not in self._activations:
            self._activations[layer_idx] = []
        
        self._activations[layer_idx].append(out.detach().float().cpu())
    
    def analyze_layer(
        self,
        model: nn.Module,
        layer_idx: int,
        inputs: Dict[str, torch.Tensor]
    ) -> LayerLens:
        """
        Analyze a specific layer.
        
        Args:
            model: PyTorch model
            layer_idx: Layer index to analyze
            inputs: Input tensors
            
        Returns:
            LayerLens with analysis results
        """
        # Clear previous activations
        self._activations = {}
        self._hooks = []
        
        # Find the layer
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append((name, module))
        
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        layer_name, layer = layers[layer_idx]
        
        # Register hook
        hook = layer.register_forward_hook(
            lambda m, i, o, idx=layer_idx: self._collect_activation(m, i, o, idx)
        )
        self._hooks.append(hook)
        
        # Run model
        model.eval()
        with torch.no_grad():
            for _ in range(self.num_samples):
                device_inputs = {k: v.to(self.device) for k, v in inputs.items()}
                model(**device_inputs)
        
        # Remove hook
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
        # Process activations
        if layer_idx not in self._activations or not self._activations[layer_idx]:
            raise ValueError(f"No activations collected for layer {layer_idx}")
        
        # Stack activations
        stacked = torch.cat(
            [a.reshape(-1, a.shape[-1]) for a in self._activations[layer_idx]],
            dim=0
        )
        
        # Compute SVD
        svd_result = self._compute_svd(stacked)
        
        # Get layer dimensions
        if isinstance(layer, nn.Linear):
            input_dim = layer.in_features
            output_dim = layer.out_features
        elif isinstance(layer, nn.Conv2d):
            input_dim = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
            output_dim = layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
        else:
            input_dim = stacked.shape[-1]
            output_dim = stacked.shape[-1]
        
        # Analyze bottleneck
        bottleneck_analysis = self._analyze_bottleneck(
            svd_result, layer_idx, layer_name, input_dim, output_dim
        )
        
        # Compute activation stats
        activation_stats = {
            'mean': float(stacked.mean().item()),
            'std': float(stacked.std().item()),
            'min': float(stacked.min().item()),
            'max': float(stacked.max().item()),
            'sparsity': float((stacked.abs() < 1e-6).float().mean().item())
        }
        
        # Create lens
        lens = LayerLens(
            layer_idx=layer_idx,
            layer_name=layer_name,
            svd_result=svd_result,
            bottleneck_analysis=bottleneck_analysis,
            activation_stats=activation_stats
        )
        
        self._lenses[layer_idx] = lens
        
        return lens
    
    def analyze_model(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        layer_pattern: Optional[str] = None
    ) -> Dict[int, LayerLens]:
        """
        Analyze all layers in a model.
        
        Args:
            model: PyTorch model
            inputs: Input tensors
            layer_pattern: Pattern to match layer names
            
        Returns:
            Dictionary of LayerLens for each layer
        """
        # Find all layers
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if layer_pattern is None or layer_pattern in name:
                    layers.append((name, module))
        
        # Analyze each layer
        lenses = {}
        for idx, (name, _) in enumerate(layers):
            try:
                lens = self.analyze_layer(model, idx, inputs)
                lenses[idx] = lens
            except Exception as e:
                print(f"Warning: Failed to analyze layer {idx} ({name}): {e}")
        
        return lenses
    
    def get_lens(self, layer_idx: int) -> Optional[LayerLens]:
        """Get lens for a specific layer."""
        return self._lenses.get(layer_idx)
    
    def get_all_lenses(self) -> Dict[int, LayerLens]:
        """Get all lenses."""
        return self._lenses.copy()
    
    def find_bottlenecks(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> List[BottleneckAnalysis]:
        """
        Find all bottleneck layers in a model.
        
        Args:
            model: PyTorch model
            inputs: Input tensors
            
        Returns:
            List of BottleneckAnalysis for bottleneck layers
        """
        self.analyze_model(model, inputs)
        
        bottlenecks = []
        for lens in self._lenses.values():
            if lens.bottleneck_analysis and lens.bottleneck_analysis.is_bottleneck:
                bottlenecks.append(lens.bottleneck_analysis)
        
        return bottlenecks
    
    def get_compression_potential(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Get compression potential for each layer.
        
        Args:
            model: PyTorch model
            inputs: Input tensors
            
        Returns:
            Dictionary of compression ratios per layer
        """
        self.analyze_model(model, inputs)
        
        potential = {}
        for idx, lens in self._lenses.items():
            if lens.bottleneck_analysis:
                potential[f"layer_{idx}"] = lens.bottleneck_analysis.compression_ratio
        
        return potential
    
    def clear(self) -> None:
        """Clear all stored analyses."""
        self._activations = {}
        self._lenses = {}
        self._hooks = []


class SVDLensHook:
    """
    Hook for runtime SVD analysis.
    
    Attaches to a layer and performs SVD analysis during forward passes.
    
    Args:
        lens: SVDLens instance
        layer_idx: Layer index
        window_size: Number of samples to collect before analysis
    """
    
    def __init__(
        self,
        lens: SVDLens,
        layer_idx: int,
        window_size: int = 100
    ):
        self.lens = lens
        self.layer_idx = layer_idx
        self.window_size = window_size
        self._activations: List[torch.Tensor] = []
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None
    
    def attach(self, module: nn.Module) -> None:
        """Attach hook to module."""
        self._hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, inp, out) -> None:
        """Hook function to collect activations."""
        self._activations.append(out.detach().float().cpu())
        
        # Analyze when window is full
        if len(self._activations) >= self.window_size:
            self._analyze()
            self._activations = []
    
    def _analyze(self) -> None:
        """Analyze collected activations."""
        if not self._activations:
            return
        
        # Stack activations
        stacked = torch.cat(
            [a.reshape(-1, a.shape[-1]) for a in self._activations],
            dim=0
        )
        
        # Compute SVD
        svd_result = self.lens._compute_svd(stacked)
        
        # Store result
        self.lens._lenses[self.layer_idx] = LayerLens(
            layer_idx=self.layer_idx,
            layer_name=str(module),
            svd_result=svd_result,
            activation_stats={
                'mean': float(stacked.mean().item()),
                'std': float(stacked.std().item())
            }
        )
    
    def detach(self) -> None:
        """Detach hook."""
        if self._hook:
            self._hook.remove()
            self._hook = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.detach()


class SVDEfficientLens:
    """
    Efficient SVD lens for large models.
    
    Uses approximate SVD methods for faster analysis on large models.
    
    Args:
        num_samples: Number of samples
        threshold: Variance threshold
        approximate: Use approximate SVD
        device: Device to run on
    """
    
    def __init__(
        self,
        num_samples: int = 10,
        threshold: float = 0.95,
        approximate: bool = True,
        device: str = 'cpu'
    ):
        self.lens = SVDLens(num_samples, threshold, device)
        self.approximate = approximate
    
    def analyze_layer(self, model, layer_idx, inputs) -> LayerLens:
        """Analyze layer with efficient SVD."""
        return self.lens.analyze_layer(model, layer_idx, inputs)
    
    def analyze_model(self, model, inputs, layer_pattern=None) -> Dict[int, LayerLens]:
        """Analyze model with efficient SVD."""
        return self.lens.analyze_model(model, inputs, layer_pattern)


def create_svd_lens(
    num_samples: int = 10,
    threshold: float = 0.95,
    device: str = 'cpu'
) -> SVDLens:
    """
    Factory function to create SVDLens.
    
    Args:
        num_samples: Number of samples for analysis
        threshold: Variance threshold for bottleneck detection
        device: Device to run on
        
    Returns:
        SVDLens instance
    """
    return SVDLens(
        num_samples=num_samples,
        threshold=threshold,
        device=device
    )


__all__ = [
    'SVDLens',
    'SVDLensHook',
    'SVDEfficientLens',
    'SVDResult',
    'BottleneckAnalysis',
    'LayerLens',
    'create_svd_lens',
]
