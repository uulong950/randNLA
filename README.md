# RandNLA Whitening Engine (`inv_sqrt_yan`)
![a319cdb74df88625c4dc3a9749cad5f6](https://github.com/user-attachments/assets/ad6fa5ae-c769-4309-9e84-74ac44473c68)


**Author:** Zhang Xiaolong (Lord of the Manifolds / 流形之主：张小龙)

A high-performance PyTorch implementation for computing the inverse square root of a symmetric positive-definite matrix ($M^{-1/2}$). By leveraging Randomized Numerical Linear Algebra (RandNLA) and the Nyström method, this algorithm completely bypasses traditional full spectral decomposition, reducing computational complexity from the standard $O(N^3)$ to $O(N^2 k)$.

This engine provides extreme acceleration for large-scale matrices commonly found in Second-Order Deep Learning Optimization (K-FAC), Fully Homomorphic Encryption (FHE) whitening, and High-Frequency Trading covariance mapping.

## Mathematical Proof

The rigorous algebraic derivation of the Nyström-based manifold sketching and the subspace projection used in this engine has been documented separately.

Please refer to **[`math.md`](math.md)** for the complete mathematical proof and geometric interpretation.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy (Required for CPU baseline benchmarking only)

## Quick Start

```python
import torch
from randnla_yan import inv_sqrt_yan

# Define target dimensions
N, K = 2048, 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct a simulated covariance matrix: M ≈ I + L * L^T
I_N = torch.eye(N, device=device)
low_rank_base = torch.randn(N, K, device=device)
M = I_N + 0.1 * (low_rank_base @ low_rank_base.T)

# Compute M^{-1/2} instantly
M_inv_sqrt = inv_sqrt_yan(M, k=K)
```

## Benchmarks & Performance

Traditional $O(N^3)$ algorithms (like SciPy's `sqrtm` and `inv`) suffer from severe performance degradation, cache pollution, and thermal throttling as matrix dimensions scale and sustained loads increase. The `inv_sqrt_yan` engine maintains stable, low-millisecond execution times regardless of continuous throughput.

### Isolated Physical Environment Benchmark (CPU)(AMD 9950X)

Tested on standard consumer-grade CPU architecture under 100 continuous iterations, eliminating L3 cache interference between baseline and the optimized engine.

| Matrix Size (N) | Sketch Size (K) | SciPy Baseline (Avg) | Yan Engine (Avg) | Absolute Physical Limit | Absolute Acceleration | Max Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1024x1024 | 64 | ~465.34 ms | ~1.51 ms | 1.42ms | **~308.1x** | 1.16e-02 |
| 2048x2048 | 64 | ~2140.42 ms | ~5.60 ms | 4.68 ms | **~382.2x** | 7.31e-03 |

> **Note:** The discrepancy between 1024x1024 and 2048x2048 performance scaling highlights cache miss penalties vs pure computational complexity. The 2048 scale demonstrates the true theoretical capability of the algorithm after bypassing memory I/O bottlenecks.
