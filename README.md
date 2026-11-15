# SVD-Lens

[![CI](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/SVD-Lens/actions/workflows/codeql.yml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

SVD-based lens on layer activations: rank, explained variance, bottleneck detection, compression potential.

## What it does

SVDLens collects a layer's activations, decomposes them with SVD, and
reports rank, effective rank, explained variance, and a bottleneck
flag (top 10% of components covering more than a variance threshold).
SVDEfficientLens uses approximate SVD for larger layers; SVDLensHook
monitors a layer at runtime over a sliding window.

## Installation

    pip install torch numpy pytest

## Usage

    from svd_lens import SVDLens

    lens = SVDLens(num_samples=10, threshold=0.95)
    lenses = lens.analyze_model(model, inputs)
    for idx, layer_lens in lenses.items():
        print(idx, layer_lens.bottleneck_analysis.rank,
              layer_lens.bottleneck_analysis.is_bottleneck)

    potential = lens.get_compression_potential(model, inputs)

## API

- SVDLens(num_samples=10, threshold=0.95): analyze_layer,
  analyze_model, find_bottlenecks, get_compression_potential, get_lens,
  get_all_lenses, clear
- SVDLensHook(lens, layer_idx, window_size): attach / detach
- SVDEfficientLens(num_samples, threshold, approximate=True)
- create_svd_lens(...) factory
- Data classes: SVDResult, BottleneckAnalysis, LayerLens

## Honest results and limitations

- "Compression potential" is the inverse of the effective rank
  fraction - a spectral estimate. No experiment here measures actual
  post-compression accuracy, so treat it as an upper-bound hint,
  not a result.
- With the default 10 samples, activation matrices are tiny and their
  spectra are sample-noisy; ranks and bottleneck flags move between
  runs. Use more samples for stable estimates.
- Earlier revisions quoted per-layer millisecond timings without
  experiments; speed is unmeasured here.
- Runtime hooks run SVD inside the forward pass - expect a noticeable
  slowdown while attached.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that spectral compressibility did
  not translate into safe compression without retraining, and the
  related carrier-mode recall failed about half of noise trials. The
  robust design kept in the toolkit is the Reed-Solomon slot lattice
  (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
