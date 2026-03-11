import torch
import numpy as np
import time
from scipy.linalg import sqrtm, inv

# ==============================================================================
# Randomized Numerical Linear Algebra (RandNLA) for Inverse Square Root Matrix
# Author: Zhang Xiaolong (Lord of the Manifolds / 流形之主：张小龙)
#
# Core Concept: Nyström-based manifold sketching to approximate M^{-1/2}
# Time Complexity: O(N^2 * k) [Optimized from traditional O(N^3) algorithms]
# ==============================================================================

def inv_sqrt_yan(M: torch.Tensor, k: int = 64, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the inverse square root of a symmetric positive-definite matrix M.
    Uses randomized sketching to project the matrix into a lower-dimensional 
    subspace, significantly reducing computational complexity.
    
    Args:
        M (torch.Tensor): Symmetric positive-definite matrix of shape (N, N).
                          Expected structure: M ≈ I + L * L^T.
        k (int): Sketch size (k << N). Controls the rank of the approximation.
        eps (float): Small constant to ensure numerical stability during division.
        
    Returns:
        torch.Tensor: Approximated inverse square root matrix M^{-1/2}.
    """
    N = M.shape[0]
    device = M.device
    dtype = M.dtype

    # 1. Generate Gaussian random matrix for subspace sketching
    Omega = torch.randn(N, k, device=device, dtype=dtype)
    Y = M @ Omega

    # 2. Orthonormalize the sketched matrix via QR decomposition
    Q, _ = torch.linalg.qr(Y)
    
    # 3. Project M into the k-dimensional subspace
    B = Q.T @ M @ Q

    # 4. Perform eigendecomposition in the reduced k x k space
    eigenvalues, V = torch.linalg.eigh(B)
    
    # 5. Compute the inverse square root in the reduced space
    B_inv_sqrt = V @ torch.diag(1.0 / torch.sqrt(torch.clamp(eigenvalues, min=eps))) @ V.T

    # 6. Reconstruct the N x N matrix using the Nyström method
    I_N = torch.eye(N, device=device, dtype=dtype)
    I_K = torch.eye(k, device=device, dtype=dtype)
    
    M_inv_sqrt = I_N + Q @ (B_inv_sqrt - I_K) @ Q.T

    return M_inv_sqrt

# ==============================================================================
# Benchmark Runner
# ==============================================================================
if __name__ == "__main__":
    # Parameters
    N = 2048
    K = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- RandNLA Whitening Benchmark ---")
    print(f"Device: {device} | Matrix Size: {N}x{N} | Sketch Size: {K}")
    
    # Generate test matrix: M = I + LowRank
    I_N = torch.eye(N, device=device, dtype=torch.float32)
    low_rank_base = torch.randn(N, K, device=device, dtype=torch.float32)
    M_gpu = I_N + 0.1 * (low_rank_base @ low_rank_base.T)
    M_cpu = M_gpu.cpu().numpy()

    # Baseline: SciPy CPU (O(N^3))
    start_cpu = time.perf_counter()
    M_inv_sqrt_cpu = inv(sqrtm(M_cpu))
    cpu_time = (time.perf_counter() - start_cpu) * 1000

    # Warm-up computation to compile instructions
    _ = inv_sqrt_yan(M_gpu, k=K)
    if torch.cuda.is_available(): torch.cuda.synchronize()

    # Optimized: RandNLA Tensor Algorithm (O(N^2 * k))
    start_gpu = time.perf_counter()
    M_inv_sqrt_gpu = inv_sqrt_yan(M_gpu, k=K)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start_gpu) * 1000

    # Error Calculation
    max_error = np.abs(M_inv_sqrt_cpu - M_inv_sqrt_gpu.cpu().numpy()).max()

    # Output Results
    print("\n[Results]")
    print(f"Max Absolute Error : {max_error:.6e}")
    print(f"SciPy Baseline Time: {cpu_time:.2f} ms")
    print(f"RandNLA _yan Time  : {gpu_time:.2f} ms")
    print("-----------------------------------")
