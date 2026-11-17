# SVD-Lens

SVD-Lens is a small PyTorch module for inspecting activations from `nn.Linear` and `nn.Conv2d` layers. It computes a singular value decomposition (SVD) of collected activations and reports spectral summaries and heuristic bottleneck indicators. The current repository contains a single Python implementation and tests; it is best treated as an early-stage analysis prototype, not a validated compression system.

## What is implemented

- `SVDLens` attaches a forward hook to one selected linear or convolutional layer, runs the model, and returns a `LayerLens` containing the SVD, activation statistics, and a `BottleneckAnalysis`.
- `analyze_model`, `find_bottlenecks`, and `get_compression_potential` provide multi-layer wrappers and summaries. The bottleneck flag uses a configurable cumulative-variance threshold and a top-10%-of-components heuristic.
- `SVDLensHook` collects activations in batches of forward calls and stores SVD results when its configured window fills.
- `SVDResult`, `BottleneckAnalysis`, and `LayerLens` are result data classes; `create_svd_lens` is a convenience constructor.

The reported compression ratio is a spectral heuristic. This code does not compress model weights, evaluate downstream accuracy, or establish that a proposed compression is safe.

## Requirements and installation

Python 3.10 or later is declared in `pyproject.toml`. Runtime dependencies are PyTorch 2.0 or later and NumPy 1.24 or later; the checked-in `requirements.txt` also includes pytest.

From a checkout, install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

The implementation is the top-level `svd_lens.py` module, so run examples from the repository root (or otherwise make that file available on `PYTHONPATH`). There is no command-line interface. Although `pyproject.toml` contains project metadata, its setuptools configuration discovers packages named `svd_lens*`, while this repository has a module file rather than a `svd_lens/` package directory; this README does not assume that installing the project itself makes the module importable.

## Example

This example uses a single tensor input with a simple sequential model, matching the input style exercised by the tests:

```python
import torch
from torch import nn
from svd_lens import SVDLens

model = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
inputs = {"input": torch.randn(16, 8)}

lens = SVDLens(num_samples=2, threshold=0.95)
result = lens.analyze_layer(model, layer_idx=0, inputs=inputs)

print(result.layer_name)
print(result.bottleneck_analysis)
print(result.activation_stats)
```

`layer_idx` is zero-based among the `nn.Linear` and `nn.Conv2d` modules found in `model.named_modules()`. For models with multiple inputs, the dictionary keys are passed as keyword arguments to the model. The model and the inputs must be on compatible devices; `device` controls where the inputs are moved, not where the model is moved.

## Tests and CI

On the repository's current commit (`4df93075f1a63f983881337476a9014d528f93dd`), `python -m pytest -q` was run locally with Python 3.12.3 and completed with **24 passed**. The workflow in `.github/workflows/ci.yml` configures pytest, compilation, and an import smoke check on Python 3.10, 3.11, and 3.12. Those workflow results were not independently checked here; the local result above is not a claim that every CI matrix job has passed.

Run the test suite from the repository root with:

```bash
python -m pytest -q
```

## Limitations

- `num_samples` repeats the same supplied input tensor(s); it does not draw new examples or consume a data loader. The resulting spectrum therefore reflects the provided batch, not an independently sampled dataset.
- The analysis selects only `nn.Linear` and `nn.Conv2d` modules. Activations are reshaped using their final tensor dimension as the feature dimension; convolutional activation interpretation is therefore limited, and the reported convolution dimensions are based on a simple channel/kernel calculation.
- `SVDEfficientLens` delegates to `SVDLens`; its `approximate` option does not select an approximate decomposition in the current implementation.
- `SVDLensHook` performs an ordinary SVD when a window fills (not an approximate SVD), stores the SVD and basic mean/std values, and resets its collected activations. It does not generate the full bottleneck analysis. Hooks add work during forward passes.
- `analyze_model` catches per-layer exceptions, prints a warning, and continues, so callers should check which layers were actually returned. When `layer_pattern` filters layers, the filtered index is passed to a method that indexes the unfiltered layer list; filtered results may consequently analyze a different layer than the filter match.
- `analyze_layer` switches the model to evaluation mode and does not restore its prior training mode. Callers managing training state should restore it explicitly.
- The bottleneck and compression values are implementation-defined summaries, not benchmark results, accuracy measurements, or a validated compression prescription. No performance benchmarks or research conclusions are reported here.

## License

The repository's `LICENSE` file contains the MIT License, copyright (c) 2024 Kapil Dhinwa. See [LICENSE](LICENSE) for the complete terms.
