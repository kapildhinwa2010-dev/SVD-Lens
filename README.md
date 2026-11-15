# SVD-Lens

[![CI](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**SVD-based Lens for Neural Network Layer Analysis**

SVD-Lens provides SVD-based analysis tools for examining neural network layer behavior, identifying bottlenecks, and understanding knowledge representation.

## Features

- **Layer-wise SVD decomposition**: Analyze each layer's activation patterns
- **Bottleneck identification**: Find layers with high compressibility
- **Knowledge concentration analysis**: Measure how concentrated knowledge is in activations
- **Compressibility scoring**: Quantify compression potential for each layer
- **Runtime monitoring hooks**: Monitor layer behavior during training/inference
- **Efficient analysis**: Approximate SVD methods for large models

## Installation

```bash
pip install SVD-Lens

# or from source
pip install git+https://github.com/kapildhinwa2010-dev/SVD-Lens.git
```

## Dependencies

- Python 3.10+
- PyTorch 2.0+
- NumPy
- pytest (for testing)

## Usage

### Basic Analysis

```python
import torch
import torch.nn as nn
from svd_lens import SVDLens

# Create model
model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 100)
)

# Create lens
lens = SVDLens(num_samples=10, threshold=0.95)

# Analyze all layers
inputs = {'input': torch.randn(10, 100)}
lenses = lens.analyze_model(model, inputs)

# Print results
for idx, layer_lens in lenses.items():
    print(f"Layer {idx}:")
    print(f"  Rank: {layer_lens.bottleneck_analysis.rank}")
    print(f"  Compression ratio: {layer_lens.bottleneck_analysis.compression_ratio:.2f}x")
    print(f"  Is bottleneck: {layer_lens.bottleneck_analysis.is_bottleneck}")
```

### Single Layer Analysis

```python
from svd_lens import SVDLens

# Create lens
lens = SVDLens(num_samples=10)

# Analyze specific layer
layer_lens = lens.analyze_layer(model, layer_idx=0, inputs={'input': torch.randn(10, 100)})

# Get SVD results
svd_result = layer_lens.svd_result
print(f"Singular values shape: {svd_result.S.shape}")
print(f"Explained variance: {svd_result.explained_variance.sum():.2%}")
```

### Find Bottlenecks

```python
from svd_lens import SVDLens

# Create lens
lens = SVDLens(num_samples=10, threshold=0.9)

# Find bottlenecks
bottlenecks = lens.find_bottlenecks(model, inputs={'input': torch.randn(10, 100)})

# Print bottlenecks
for bottleneck in bottlenecks:
    print(f"Layer {bottleneck.layer_idx}: {bottleneck.bottleneck_strength:.2%} variance in top 10%")
```

### Runtime Monitoring

```python
import torch.nn as nn
from svd_lens import SVDLens, SVDLensHook

# Create model and lens
model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 100))
lens = SVDLens(num_samples=10)

# Create hook for layer 0
hook = SVDLensHook(lens, layer_idx=0, window_size=100)

# Attach to layer
hook.attach(model[0])

# Run model
for _ in range(1000):
    x = torch.randn(10, 100)
    output = model(x)

# Get analysis results
layer_lens = lens.get_lens(0)
print(f"Layer 0 analysis: {layer_lens}")

# Detach hook
hook.detach()
```

### Compression Potential

```python
from svd_lens import SVDLens

# Create lens
lens = SVDLens(num_samples=10, threshold=0.95)

# Get compression potential
potential = lens.get_compression_potential(model, inputs={'input': torch.randn(10, 100)})

# Print potential
for layer_name, ratio in potential.items():
    print(f"{layer_name}: {ratio:.2f}x compression potential")
```

## Analysis Metrics

### Rank
- **Full rank**: All singular values are non-zero
- **Reduced rank**: Some singular values are zero or near-zero
- **Effective rank**: Weighted by singular value magnitudes

### Compression Ratio
- **High ratio (> 4x)**: Highly compressible layer
- **Medium ratio (2-4x)**: Moderately compressible
- **Low ratio (< 2x)**: Low compressibility

### Bottleneck Detection
- **Bottleneck layer**: Top 10% of components explain > threshold variance
- **Bottleneck strength**: Proportion of variance in top components

## Performance

### Analysis Speed
- **Small layers (< 1K dim)**: ~1-2ms per layer
- **Medium layers (1K-10K dim)**: ~5-10ms per layer
- **Large layers (> 10K dim)**: ~10-50ms per layer

### Memory Usage
- **Activation storage**: ~2-4x layer size per sample
- **SVD computation**: ~O(n^2) for layer size n

## API Reference

### SVDLens

```python
class SVDLens
```

SVD-based lens for analyzing neural network layers.

**Parameters:**
- `num_samples` (int): Number of samples for analysis (default: 10)
- `threshold` (float): Variance threshold for bottleneck detection (default: 0.95)
- `device` (str): Device to run on (default: 'cpu')

**Methods:**
- `analyze_layer(model, layer_idx, inputs)`: Analyze a specific layer
- `analyze_model(model, inputs, layer_pattern=None)`: Analyze all layers
- `find_bottlenecks(model, inputs)`: Find all bottleneck layers
- `get_compression_potential(model, inputs)`: Get compression potential per layer
- `get_lens(layer_idx)`: Get lens for a specific layer
- `get_all_lenses()`: Get all lenses
- `clear()`: Clear all stored analyses

### SVDLensHook

```python
class SVDLensHook
```

Hook for runtime SVD analysis.

**Parameters:**
- `lens` (SVDLens): SVDLens instance
- `layer_idx` (int): Layer index
- `window_size` (int): Number of samples to collect before analysis (default: 100)

**Methods:**
- `attach(module)`: Attach hook to module
- `detach()`: Detach hook

### SVDEfficientLens

```python
class SVDEfficientLens
```

Efficient SVD lens for large models.

**Parameters:**
- `num_samples` (int): Number of samples (default: 10)
- `threshold` (float): Variance threshold (default: 0.95)
- `approximate` (bool): Use approximate SVD (default: True)
- `device` (str): Device to run on (default: 'cpu')

### Data Classes

#### SVDResult
Result of SVD decomposition.

**Attributes:**
- `U` (torch.Tensor): Left singular vectors
- `S` (torch.Tensor): Singular values
- `Vt` (torch.Tensor): Right singular vectors (transposed)
- `explained_variance` (torch.Tensor): Variance explained by each component
- `cumulative_variance` (torch.Tensor): Cumulative variance

#### BottleneckAnalysis
Analysis of layer bottleneck.

**Attributes:**
- `layer_idx` (int): Layer index
- `layer_name` (str): Layer name
- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension
- `rank` (int): Rank of activation matrix
- `effective_rank` (float): Effective rank
- `compression_ratio` (float): Compression ratio
- `top_k_variance` (Dict[int, float]): Variance for top k components
- `is_bottleneck` (bool): Whether layer is a bottleneck
- `bottleneck_strength` (float): Bottleneck strength (0-1)

#### LayerLens
Lens for a single layer.

**Attributes:**
- `layer_idx` (int): Layer index
- `layer_name` (str): Layer name
- `svd_result` (SVDResult, optional): SVD result
- `bottleneck_analysis` (BottleneckAnalysis, optional): Bottleneck analysis
- `activation_stats` (Dict[str, float]): Activation statistics

### Factory Function

```python
def create_svd_lens(num_samples=10, threshold=0.95, device='cpu')
```

Factory function to create SVDLens.

**Parameters:**
- `num_samples` (int): Number of samples (default: 10)
- `threshold` (float): Variance threshold (default: 0.95)
- `device` (str): Device (default: 'cpu')

**Returns:**
- `SVDLens`: SVDLens instance

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please open an issue or pull request.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [BehaviorDNA](https://github.com/kapildhinwa2010-dev/BehaviorDNA) - Behavior analysis and compression
- [ActivationCompress](https://github.com/kapildhinwa2010-dev/ActivationCompress) - Activation compression suite
- [FourierNet](https://github.com/kapildhinwa2010-dev/FourierNet) - Fourier-based weight generation
- [DNA-Encoder](https://github.com/kapildhinwa2010-dev/DNA-Encoder) - DNA-based model encoding

## Citation

If you use this library, please cite:

```bibtex
@misc{SVDLens2024,
  author = {Kapil Dhinwa},
  title = {SVD-Lens: SVD-based Lens for Neural Network Layer Analysis},
  year = {2024},
  url = {https://github.com/kapildhinwa2010-dev/SVD-Lens},
}
```
