# Rigorous Mathematical Proof: RandNLA Whitening ($M^{-1/2}$)

## 1. Theorem

Let $M \in \mathbb{R}^{N \times N}$ be a symmetric positive-definite matrix constructed as $M = I_N + L L^T$, where $L \in \mathbb{R}^{N \times K}$ is a low-rank feature matrix with $K < N$. Let $L$ have the economy-size Singular Value Decomposition (SVD) $L = Q \Sigma_L V^T$, where $Q \in \mathbb{R}^{N \times K}$ has orthonormal columns ($Q^T Q = I_K$). 
Let $B = Q^T M Q \in \mathbb{R}^{K \times K}$. Then the inverse square root of $M$ is strictly given by:

$$M^{-1/2} = I_N + Q (B^{-1/2} - I_K) Q^T$$

## 2. Proof

**Step 1: Spectral Decomposition of $M$**

Given $L = Q \Sigma_L V^T$, we can express the low-rank component as:
$$L L^T = (Q \Sigma_L V^T)(V \Sigma_L Q^T) = Q \Sigma_L^2 Q^T$$

Thus, the target matrix $M$ is:
$$M = I_N + Q \Sigma_L^2 Q^T$$

By the axioms of finite-dimensional Hilbert spaces, since $Q \in \mathbb{R}^{N \times K}$ has orthonormal columns, there exists an orthogonal complement matrix $Q_\perp \in \mathbb{R}^{N \times (N-K)}$ such that the concatenated matrix $[Q, Q_\perp] \in \mathbb{R}^{N \times N}$ is orthogonal. 

The identity matrix $I_N$ can be exactly resolved by the sum of orthogonal projectors onto these mutually exclusive subspaces:
$$I_N = Q Q^T + Q_\perp Q_\perp^T$$

Substituting this resolution of the identity into the expression for $M$:
$$M = (Q Q^T + Q_\perp Q_\perp^T) + Q \Sigma_L^2 Q^T$$
$$M = Q (I_K + \Sigma_L^2) Q^T + Q_\perp I_{N-K} Q_\perp^T$$

Since both $I_K + \Sigma_L^2$ and $I_{N-K}$ are diagonal matrices, and $[Q, Q_\perp]$ is an orthogonal matrix, this equation represents the exact eigendecomposition (Spectral Decomposition) of $M$.

**Step 2: Continuous Functional Calculus**

Let $f: \mathbb{R}^+ \to \mathbb{R}^+$ be a continuous scalar function. By the Spectral Theorem for symmetric positive-definite matrices, applying a matrix function $f(M)$ is equivalent to applying $f$ element-wise to its eigenvalues while preserving the eigenvectors:
$$f(M) = Q f(I_K + \Sigma_L^2) Q^T + Q_\perp f(I_{N-K}) Q_\perp^T$$

Let the function be the inverse square root, $f(x) = x^{-1/2}$. Applying this yields:
$$M^{-1/2} = Q (I_K + \Sigma_L^2)^{-1/2} Q^T + Q_\perp (I_{N-K})^{-1/2} Q_\perp^T$$

Since the inverse square root of an identity matrix is strictly itself ($(I_{N-K})^{-1/2} = I_{N-K}$), the expression simplifies to:
$$M^{-1/2} = Q (I_K + \Sigma_L^2)^{-1/2} Q^T + Q_\perp Q_\perp^T$$

**Step 3: Subspace Reconstruction**

Recall the identity resolution $I_N = Q Q^T + Q_\perp Q_\perp^T$. Rearranging this gives the exact definition of the null space projector:
$$Q_\perp Q_\perp^T = I_N - Q Q^T$$

Substitute this back into the equation for $M^{-1/2}$:
$$M^{-1/2} = Q (I_K + \Sigma_L^2)^{-1/2} Q^T + (I_N - Q Q^T)$$
$$M^{-1/2} = I_N + Q \left( (I_K + \Sigma_L^2)^{-1/2} - I_K \right) Q^T$$

**Step 4: Inner Core Matrix Equivalence**

Define the projection of $M$ onto the principal subspace as $B \in \mathbb{R}^{K \times K}$:
$$B = Q^T M Q$$
$$B = Q^T (I_N + Q \Sigma_L^2 Q^T) Q$$
$$B = Q^T Q + (Q^T Q) \Sigma_L^2 (Q^T Q)$$

Since $Q^T Q = I_K$, this rigorously simplifies to:
$$B = I_K + \Sigma_L^2$$

Substituting $B$ into the result from Step 3, we arrive at the final closed-form matrix equation:
$$M^{-1/2} = I_N + Q (B^{-1/2} - I_K) Q^T$$

$\blacksquare$

---

## 3. Engineering Implications: Absolute Equality ($=$) vs. Low-Rank Approximation ($\approx$)

It is crucial to distinguish between the theoretical exactness of the derivation above and its practical application in engineering scenarios (such as deep learning or high-frequency trading).

* **The Mathematical Reality (Absolute Equality):** The proof above establishes a strictly exact algebraic equality ($=$). It proves that **if** a matrix $M$ is perfectly composed of $I_N + LL^T$, the derived formula is the absolute closed-form analytical solution for its inverse square root. There are no truncated series, dropped values, or numerical estimations in this logical domain.
* **The Physical Reality (Engineering Approximation):** In the physical world, a raw covariance matrix $M_{real}$ is almost never perfectly low-rank. It contains full-rank noise: $M_{real} = I_N + LL^T + E_{noise}$. When this engine utilizes Randomized Numerical Linear Algebra (RandNLA) via Nyström sketching, it actively captures the principal components ($LL^T$) and **intentionally discards** the long-tail noise ($E_{noise}$). 

Therefore, in the codebase (`randnla_yan.py`), the computation is an approximation ($\approx$). We trade the absolute precision of the negligible noise $E_{noise}$ to bypass the full $O(N^3)$ spectral decomposition, achieving massive $O(N^2 k)$ acceleration while retaining the core manifold structure of the matrix.
