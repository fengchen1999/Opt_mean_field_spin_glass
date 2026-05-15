import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss


# ============================================================
# 1. Generate GOE matrix
# ============================================================

def sample_goe(n, seed=0):
    """
    GOE normalization close to Montanari:
    off-diagonal ~ N(0, 1/n), diagonal ~ N(0, 2/n).
    """
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, n)) / np.sqrt(n)
    A = np.triu(G, 1)
    A = A + A.T
    diag = rng.normal(scale=np.sqrt(2 / n), size=n)
    np.fill_diagonal(A, diag)
    return A


# ============================================================
# 2. Stable terminal condition Phi(1,x) = log(2 cosh x)
# ============================================================

def log_2_cosh(x):
    # log(2 cosh x) = log(exp(x)+exp(-x))
    return np.logaddexp(x, -x)


# ============================================================
# 3. Solve Parisi PDE for a given mu(t)
# ============================================================

# def solve_parisi_pde(beta, mu_vals, Nt=120, Nx=401, xmax=6.0):
#     """
#     Solve:
#         Phi_t + 0.5 beta^2 Phi_xx + 0.5 beta^2 mu(t) Phi_x^2 = 0
#         Phi(1,x) = log(2 cosh x)
#
#     Backward explicit finite-difference scheme.
#
#     Parameters
#     ----------
#     beta : float
#     mu_vals : array, length Nt+1
#         CDF values mu(t_j) on t grid. Should be nondecreasing in [0,1].
#     Nt : int
#     Nx : int
#     xmax : float
#
#     Returns
#     -------
#     t_grid, x_grid, Phi
#         Phi has shape (Nt+1, Nx), Phi[j,:] approximates Phi(t_j, x_grid).
#     """
#     t_grid = np.linspace(0.0, 1.0, Nt + 1)
#     x_grid = np.linspace(-xmax, xmax, Nx)
#     dt = t_grid[1] - t_grid[0]
#     dx = x_grid[1] - x_grid[0]
#
#     Phi = np.zeros((Nt + 1, Nx))
#     Phi[-1, :] = log_2_cosh(x_grid)
#
#     # Backward in time: from t=1 to t=0
#     for j in range(Nt - 1, -1, -1):
#         phi_next = Phi[j + 1, :].copy()
#         mu = mu_vals[j + 1]
#
#         # central differences
#         phi_x = np.zeros_like(phi_next)
#         phi_xx = np.zeros_like(phi_next)
#
#         phi_x[1:-1] = (phi_next[2:] - phi_next[:-2]) / (2 * dx)
#         phi_xx[1:-1] = (phi_next[2:] - 2 * phi_next[1:-1] + phi_next[:-2]) / dx**2
#
#         # boundary: approximate derivative using terminal asymptotic behavior
#         # For large |x|, Phi_x approx sign(x), Phi_xx approx 0.
#         phi_x[0] = -1.0
#         phi_x[-1] = 1.0
#         phi_xx[0] = 0.0
#         phi_xx[-1] = 0.0
#
#         rhs = 0.5 * beta**2 * phi_xx + 0.5 * beta**2 * mu * phi_x**2
#
#         # Since PDE is Phi_t + rhs = 0,
#         # going backward: Phi(t-dt) = Phi(t) + dt * rhs
#         Phi[j, :] = phi_next + dt * rhs
#
#         # crude boundary extrapolation
#         Phi[j, 0] = Phi[j, 1] + (x_grid[0] - x_grid[1]) * (-1.0)
#         Phi[j, -1] = Phi[j, -2] + (x_grid[-1] - x_grid[-2]) * (1.0)
#
#     return t_grid, x_grid, Phi

def solve_parisi_pde(beta, mu_vals, Nt=120, Nx=501, xmax=8.0, gh_order=40):
    """
    Stable Parisi PDE solver using Gaussian convolution for stepwise mu.

    PDE:
        Phi_t + 0.5 beta^2 Phi_xx + 0.5 beta^2 mu(t) Phi_x^2 = 0
        Phi(1,x) = log(2 cosh x)

    For constant mu=m on one time interval:
        Phi_old(x) = 1/m log E exp(m Phi_new(x + beta sqrt(dt) Z))
    if m=0:
        Phi_old(x) = E Phi_new(x + beta sqrt(dt) Z)
    """
    t_grid = np.linspace(0.0, 1.0, Nt + 1)
    x_grid = np.linspace(-xmax, xmax, Nx)
    dt = t_grid[1] - t_grid[0]

    Phi = np.zeros((Nt + 1, Nx))
    Phi[-1, :] = log_2_cosh(x_grid)

    # Gauss-Hermite quadrature for Z ~ N(0,1)
    gh_x, gh_w = hermgauss(gh_order)
    z_nodes = np.sqrt(2.0) * gh_x
    z_weights = gh_w / np.sqrt(np.pi)

    for j in range(Nt - 1, -1, -1):
        m = float(mu_vals[j + 1])
        phi_next = Phi[j + 1, :]

        interp_phi = interp1d(
            x_grid,
            phi_next,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )

        sigma = beta * np.sqrt(dt)
        phi_old = np.empty_like(x_grid)

        for ix, x in enumerate(x_grid):
            y = x + sigma * z_nodes
            vals = interp_phi(y)

            if abs(m) < 1e-12:
                phi_old[ix] = np.sum(z_weights * vals)
            else:
                # log E exp(m vals)
                phi_old[ix] = logsumexp(m * vals, b=z_weights) / m

        Phi[j, :] = phi_old

    return t_grid, x_grid, Phi


def compute_derivatives(Phi, x_grid):
    """
    Compute Phi_x and Phi_xx on grid.
    """
    dx = x_grid[1] - x_grid[0]
    Phi_x = np.zeros_like(Phi)
    Phi_xx = np.zeros_like(Phi)

    Phi_x[:, 1:-1] = (Phi[:, 2:] - Phi[:, :-2]) / (2 * dx)
    Phi_xx[:, 1:-1] = (Phi[:, 2:] - 2 * Phi[:, 1:-1] + Phi[:, :-2]) / dx**2

    Phi_x[:, 0] = -1.0
    Phi_x[:, -1] = 1.0
    Phi_xx[:, 0] = 0.0
    Phi_xx[:, -1] = 0.0

    return Phi_x, Phi_xx


# ============================================================
# 4. Parametrize monotone mu(t)
# ============================================================

def theta_to_mu(theta, Nt):
    """
    Convert unconstrained theta to a monotone CDF mu(t) in [0,1].
    We use softmax increments so that mu(0) >= 0, mu(1)=1.
    """
    w = np.exp(theta - np.max(theta))
    w = w / np.sum(w)
    mu = np.cumsum(w)
    mu = np.concatenate([[0.0], mu])
    # interpolate from len(theta)+1 to Nt+1 grid
    coarse_t = np.linspace(0, 1, len(mu))
    fine_t = np.linspace(0, 1, Nt + 1)
    mu_fine = np.interp(fine_t, coarse_t, mu)
    mu_fine[-1] = 1.0
    return mu_fine


def parisi_functional(beta, theta, Nt=120, Nx=401, xmax=6.0):
    """
    Approximate P_beta(mu).
    """
    mu_vals = theta_to_mu(theta, Nt)
    t_grid, x_grid, Phi = solve_parisi_pde(beta, mu_vals, Nt=Nt, Nx=Nx, xmax=xmax)

    # Phi(0,0)
    phi00 = np.interp(0.0, x_grid, Phi[0, :])

    # integral int_0^1 t mu(t) dt
    integral = np.trapz(t_grid * mu_vals, t_grid)

    P = phi00 - 0.5 * beta**2 * integral
    return P


def optimize_mu(beta, Nt=120, Nx=401, xmax=6.0, n_blocks=20, maxiter=100):
    """
    Approximate minimizer mu_beta by optimizing a monotone step CDF.
    """
    theta0 = np.zeros(n_blocks)

    def obj(theta):
        return parisi_functional(beta, theta, Nt=Nt, Nx=Nx, xmax=xmax)

    res = minimize(
        obj,
        theta0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-3, "disp": True},
    )

    mu_vals = theta_to_mu(res.x, Nt)
    t_grid, x_grid, Phi = solve_parisi_pde(beta, mu_vals, Nt=Nt, Nx=Nx, xmax=xmax)
    Phi_x, Phi_xx = compute_derivatives(Phi, x_grid)

    return {
        "result": res,
        "t_grid": t_grid,
        "x_grid": x_grid,
        "mu_vals": mu_vals,
        "Phi": Phi,
        "Phi_x": Phi_x,
        "Phi_xx": Phi_xx,
        "P_beta": res.fun,
    }


# ============================================================
# 5. Build interpolation oracles for Phi_x and Phi_xx
# ============================================================

class ParisiOracle:
    def __init__(self, t_grid, x_grid, mu_vals, Phi_x, Phi_xx):
        self.t_grid = t_grid
        self.x_grid = x_grid
        self.mu_vals = mu_vals

        self.interp_Phi_x = RegularGridInterpolator(
            (t_grid, x_grid), Phi_x, bounds_error=False, fill_value=None
        )
        self.interp_Phi_xx = RegularGridInterpolator(
            (t_grid, x_grid), Phi_xx, bounds_error=False, fill_value=None
        )

    def mu(self, t):
        return np.interp(t, self.t_grid, self.mu_vals)

    def phix(self, t, x):
        x_clip = np.clip(x, self.x_grid[0], self.x_grid[-1])
        pts = np.column_stack([np.full_like(x_clip, t, dtype=float), x_clip])
        return self.interp_Phi_x(pts)

    def phixx(self, t, x):
        x_clip = np.clip(x, self.x_grid[0], self.x_grid[-1])
        pts = np.column_stack([np.full_like(x_clip, t, dtype=float), x_clip])
        return self.interp_Phi_xx(pts)


# ============================================================
# 6. IAMP algorithm, simplified Appendix B style
# ============================================================

def round_sequential(A, z):
    """
    Algorithm 2 style rounding:
    1. project to [-1,1]^n
    2. sequentially set sign according to local field
    """
    zt = np.clip(z.copy(), -1.0, 1.0)
    n = len(zt)

    for i in range(n):
        h_i = A[i, :] @ zt - A[i, i] * zt[i]
        zt[i] = 1.0 if h_i >= 0 else -1.0

    return zt.astype(int)


def iamp_sk(A, beta, delta, oracle, q_star=None, M=8.0, seed=1):
    """
    Simplified IAMP algorithm for SK Hamiltonian.

    This follows the pseudo-code structure:
        u^{k+1} = A(g^{k-1} * u^k) - b_k g^{k-2} * u^{k-1}
        x^k = x^{k-1} + beta^2 mu(k delta) Phi_x(k delta, x^{k-1}) delta
              + beta sqrt(delta) u^k
        g^k = sqrt(n) Phi_xx(k delta, x^k) / ||Phi_xx(k delta, x^k)||

    Notes:
    - This is a numerical approximation.
    - The paper includes technical truncation and oracle assumptions.
    """
    rng = np.random.default_rng(seed)
    n = A.shape[0]

    if q_star is None:
        # crude numerical definition: largest t where mu(t) < 1 - tol
        tol = 1e-3
        idx = np.where(oracle.mu_vals < 1.0 - tol)[0]
        if len(idx) == 0:
            q_star = 1.0
        else:
            q_star = oracle.t_grid[idx[-1]]

    K = int(np.floor(q_star / delta))
    u_prev = np.zeros(n)
    u = rng.normal(size=n)
    x = np.zeros(n)

    g_minus2 = np.zeros(n)
    g_minus1 = np.ones(n)
    b = 0.0

    g_list = []
    u_list = []

    for k in range(K):
        # AMP update
        f = g_minus1 * np.clip(u, -M, M)
        onsager = b * g_minus2 * np.clip(u_prev, -M, M)
        u_next = A @ f - onsager

        # State update x
        t = min(k * delta, 1.0)
        mu_t = oracle.mu(t)
        phix = oracle.phix(t, x)
        x_new = (
            x
            + beta**2 * mu_t * phix * delta
            + beta * np.sqrt(delta) * np.clip(u, -M, M)
        )

        # g update
        phixx = oracle.phixx(t, x_new)
        norm = np.linalg.norm(phixx)
        if norm < 1e-12:
            g_new = np.zeros(n)
        else:
            g_new = np.sqrt(n) * phixx / norm

        # b_{k+1}
        b_new = np.mean(g_new)

        g_list.append(g_minus1.copy())
        u_list.append(u.copy())

        # shift variables
        u_prev, u = u, u_next
        x = x_new
        g_minus2, g_minus1 = g_minus1, g_new
        b = b_new

    # z = sqrt(delta) sum_{k=1}^K g^{k-1} * u^k
    z = np.zeros(n)
    for gk, uk in zip(g_list, u_list):
        z += np.sqrt(delta) * gk * np.clip(uk, -M, M)

    sigma = round_sequential(A, z)
    return sigma, z


def sk_energy(A, sigma):
    return sigma @ A @ sigma

def get_theoretical_mu(beta, Nt):
    """
    Constructs the linear mu(t) based on the low-temp limit theory.
    mu(t) approx t / beta for t in [0, 1).
    Note: mu(1) must be 1.0 by definition.
    """
    t_grid = np.linspace(0, 1, Nt + 1)
    mu_vals = t_grid / beta
    mu_vals[-1] = 1.0  # Crucial jump to 1.0 at the boundary
    return mu_vals

# ============================================================
# 7. Example run
# ============================================================

if __name__ == "__main__":
    if __name__ == "__main__":
        n = 1000  # Larger n reduces finite-size noise
        beta = 20.0  # Higher beta closer to ground state
        delta = 0.01  # Finer time steps for IAMP

        print(f"Generating GOE matrix (n={n})...")
        A = sample_goe(n, seed=123)

        print(f"Setting mu(t) = t/{beta} (Low-temp approximation)...")
        Nt = 200
        mu_vals = get_theoretical_mu(beta, Nt)

        # Solve PDE with higher resolution
        # Increased xmax and Nx to prevent boundary leakage at high beta
        t_grid, x_grid, Phi = solve_parisi_pde(
            beta=beta,
            mu_vals=mu_vals,
            Nt=Nt,
            Nx=601,
            xmax=10.0,
            gh_order=40
        )

        Phi_x, Phi_xx = compute_derivatives(Phi, x_grid)

        oracle = ParisiOracle(t_grid, x_grid, mu_vals, Phi_x, Phi_xx)

        print("Running IAMP...")
        # Using M=10.0 and q_star=1.0 for the full path
        sigma, z = iamp_sk(A, beta=beta, delta=delta, oracle=oracle, q_star=1.0, M=10.0, seed=321)

        E = sk_energy(A, sigma)
        h_n = E / (2 * n)
        print("-" * 30)
        print(f"Final Hamiltonian H/n: {h_n:.5f}")
        print(f"Target Baseline: ~0.7632")
        print("-" * 30)

# if __name__ == "__main__":
#
# n = 500
#
# beta = 10.0
#
# delta = 0.02
#
# print("Generating GOE matrix...")
#
# A = sample_goe(n, seed=123)
#
# print("Approximating Parisi measure and PDE solution...")
#
# # For demo, use modest grid. Increase Nt, Nx, n_blocks, maxiter for better accuracy.
#
# sol = optimize_mu(
#
#     beta=beta,
#
#     Nt=80,
#
#     Nx=301,
#
#     xmax=6.0,
#
#     n_blocks=12,
#
#     maxiter=40,
#
# )
#
# oracle = ParisiOracle(
#
#     sol["t_grid"],
#
#     sol["x_grid"],
#
#     sol["mu_vals"],
#
#     sol["Phi_x"],
#
#     sol["Phi_xx"],
#
# )
#
# print("Approximate P_beta:", sol["P_beta"])
#
# print("Running IAMP...")
#
# sigma, z = iamp_sk(A, beta=beta, delta=delta, oracle=oracle, M=6.0, seed=321)
#
# E = sk_energy(A, sigma)
#
# print("sigma energy <sigma,A sigma> / n =", E / n)
#
# print("Hamiltonian H/n = <sigma,A sigma>/(2n) =", E / (2 * n))