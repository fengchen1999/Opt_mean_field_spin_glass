import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "font.family": "Times New Roman"
})


# ============================================================
# 1. GOE matrix
# ============================================================
def sample_goe(n, seed=0):
    rng = np.random.default_rng(seed)

    M = rng.normal(size=(n, n)) / np.sqrt(n)
    G = np.triu(M, 1)
    G = G + G.T

    diag = rng.normal(scale=np.sqrt(2.0 / n), size=n)
    np.fill_diagonal(G, diag)

    return G


# ============================================================
# 2. Objective and gradient
# ============================================================
def H_quad(G, m):
    return 0.5 * float(m @ G @ m)


def grad_quad(G, m):
    return G @ m


# ============================================================
# 3. Spectral solution
# ============================================================
def spectral_solution(G, mode="max"):
    """
    For H(m)=0.5 m^T G m, ||m||^2=N:

    max solution:
        m = sqrt(N) * v_max(G)

    min solution:
        m = sqrt(N) * v_min(G)
    """
    n = G.shape[0]
    eigvals, eigvecs = np.linalg.eigh(G)

    if mode == "max":
        lam = eigvals[-1]
        v = eigvecs[:, -1]
    elif mode == "min":
        lam = eigvals[0]
        v = eigvecs[:, 0]
    else:
        raise ValueError("mode must be 'max' or 'min'.")

    m_spec = np.sqrt(n) * v
    E_spec = H_quad(G, m_spec) / n

    return m_spec, lam, E_spec


# ============================================================
# 4. Projection
# ============================================================
def project_orthogonal(m, x):
    norm2 = float(m @ m)
    if norm2 < 1e-14:
        return x.copy()
    return x - m * ((m @ x) / norm2)


def projected_hessian_matrix(G, m):
    n = len(m)
    norm2 = float(m @ m)

    if norm2 < 1e-14:
        P = np.eye(n)
    else:
        P = np.eye(n) - np.outer(m, m) / norm2

    Hperp = P @ G @ P
    return 0.5 * (Hperp + Hperp.T)


# ============================================================
# 5. Hessian ascent
# ============================================================
def hessian_ascent_spherical_goe(
    G,
    delta=0.02,
    mode="max",
    seed=1,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    n = G.shape[0]

    if mode not in ["max", "min"]:
        raise ValueError("mode must be 'max' or 'min'.")

    K = int(round(1.0 / delta))
    delta = 1.0 / K

    # Initial point: ||m||^2 = N delta
    u = rng.normal(size=n)
    u = u / np.linalg.norm(u)
    m = np.sqrt(n * delta) * u

    m_path = [m.copy()]
    q_hist = [float(m @ m / n)]
    E_hist = [H_quad(G, m) / n]

    if verbose:
        print(f"n={n}, K={K}, delta={delta:.5f}, mode={mode}")
        print(f"initial m={m}")
        print(f"initial q={q_hist[-1]:.6f}, H/N={E_hist[-1]:.6f}")

    for k in range(1, K):
        Hperp = projected_hessian_matrix(G, m)

        eigvals, eigvecs = np.linalg.eigh(Hperp)

        if mode == "max":
            idx = np.argmax(eigvals)
        else:
            idx = np.argmin(eigvals)

        v = eigvecs[:, idx]

        # Ensure v is orthogonal to m
        v = project_orthogonal(m, v)
        v_norm = np.linalg.norm(v)

        if v_norm < 1e-12:
            v = rng.normal(size=n)
            v = project_orthogonal(m, v)
            v = v / np.linalg.norm(v)
        else:
            v = v / v_norm

        # Choose sign for first-order improvement
        g = grad_quad(G, m)

        if mode == "max":
            if v @ g < 0:
                v = -v
        else:
            if v @ g > 0:
                v = -v

        # Orthogonal update
        m = m + np.sqrt(n * delta) * v

        m_path.append(m.copy())
        q_hist.append(float(m @ m / n))
        E_hist.append(H_quad(G, m) / n)

        if verbose:
            print(
                f"k={k:3d}, "
                f"m={m}, "
                f"q={q_hist[-1]:.6f}, "
                f"H/N={E_hist[-1]:.6f}"
            )

    hist = {
        "m_path": np.array(m_path),
        "q": np.array(q_hist),
        "energy": np.array(E_hist),
        "delta": delta,
        "K": K,
    }

    return m, hist


# ============================================================
# 6. Plot 2D trajectory with spectral solution
# ============================================================
def plot_trajectory_2d(hist, n, m_spec, title="2D trajectory of m"):
    m_path = hist["m_path"]

    theta = np.linspace(0, 2 * np.pi, 400)
    radius = np.sqrt(n)

    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)

    plt.figure(figsize=(7, 7))

    # Final feasible circle
    plt.plot(circle_x, circle_y, linestyle="--", label=r"$\|m\|=\sqrt{N}$")

    # Hessian trajectory
    plt.plot(
        m_path[:, 0],
        m_path[:, 1],
        marker="o",
        label="Hessian trajectory",
    )

    # Start and final
    plt.scatter(
        m_path[0, 0],
        m_path[0, 1],
        s=100,
        marker="o",
        label="start",
    )

    plt.scatter(
        m_path[-1, 0],
        m_path[-1, 1],
        s=100,
        marker="s",
        label="Hessian final",
    )

    # Spectral solution
    plt.scatter(
        m_spec[0],
        m_spec[1],
        s=180,
        marker="*",
        label="spectral solution",
    )

    # Also plot opposite spectral point for reference
    plt.scatter(
        -m_spec[0],
        -m_spec[1],
        s=100,
        marker="x",
        label="opposite spectral point",
    )

    # Annotate selected path points
    for idx in range(len(m_path)):
        if idx % max(1, len(m_path)//10) == 0 or idx == len(m_path)-1:
            plt.annotate(str(idx), (m_path[idx, 0], m_path[idx, 1]))

    plt.xlabel(r"$m_1$")
    plt.ylabel(r"$m_2$")
    # plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "./Results/path_2D.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()


# ============================================================
# 7. Plot 3D trajectory with spectral solution
# ============================================================
def plot_trajectory_3d(hist, n, m_spec, title="3D trajectory of m"):
    m_path = hist["m_path"]
    radius = np.sqrt(n)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Sphere surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)

    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(xs, ys, zs, alpha=0.15, linewidth=0)

    # Hessian trajectory
    ax.plot(
        m_path[:, 0],
        m_path[:, 1],
        m_path[:, 2],
        marker="o",
        label="Hessian trajectory",
    )

    # Start
    ax.scatter(
        m_path[0, 0],
        m_path[0, 1],
        m_path[0, 2],
        s=100,
        marker="o",
        label="start",
    )

    # Hessian final
    ax.scatter(
        m_path[-1, 0],
        m_path[-1, 1],
        m_path[-1, 2],
        s=100,
        marker="s",
        label="Hessian final",
    )

    # Spectral solution
    ax.scatter(
        m_spec[0],
        m_spec[1],
        m_spec[2],
        s=180,
        marker="*",
        label="spectral solution",
    )

    # Opposite spectral point
    ax.scatter(
        -m_spec[0],
        -m_spec[1],
        -m_spec[2],
        s=100,
        marker="x",
        label="opposite spectral point",
    )

    # Annotate selected points
    # for idx in range(len(m_path)):
    #     if idx % max(1, len(m_path)//10) == 0 or idx == len(m_path)-1:
    #         ax.text(
    #             m_path[idx, 0],
    #             m_path[idx, 1],
    #             m_path[idx, 2],
    #             str(idx),
    #         )

    ax.set_xlabel(r"$m_1$")
    ax.set_ylabel(r"$m_2$")
    ax.set_zlabel(r"$m_3$")
    ax.set_title(title)
    ax.legend()

    ax.set_box_aspect([1, 1, 1])
    lim = radius * 1.1
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])

    plt.tight_layout()
    plt.savefig(
        "./Results/path_3D.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()



# ============================================================
# 8. Plot energy trajectory
# ============================================================
def plot_energy(hist, E_spec=None):
    plt.figure(figsize=(7, 5))
    plt.plot(hist["q"], hist["energy"], marker="o", label="Hessian trajectory")

    if E_spec is not None:
        plt.axhline(
            y=E_spec,
            linestyle="--",
            label="spectral optimum",
        )

    plt.xlabel(r"$q=\|m\|^2/N$")
    plt.ylabel(r"$H(m)/N$")
    plt.title(r"Energy trajectory: $H(m)/N$ vs $q$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "./Results/path.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()
    plt.show()


# ============================================================
# 9. Main
# ============================================================
if __name__ == "__main__":
    # Choose n = 2 or n = 3
    n = 200

    delta = 0.02
    mode = "max"        # "max" or "min"
    seed_matrix = 42
    seed_alg = 1

    G = sample_goe(n, seed=seed_matrix)

    print("GOE matrix G:")
    print(G)

    # Spectral solution
    m_spec, lam_spec, E_spec = spectral_solution(G, mode=mode)

    print("\nSpectral solution:")
    print("lambda =", lam_spec)
    print("m_spec =", m_spec)
    print("||m_spec||^2 / N =", (m_spec @ m_spec) / n)
    print("H(m_spec)/N =", E_spec)

    # Hessian ascent/descent
    m_final, hist = hessian_ascent_spherical_goe(
        G,
        delta=delta,
        mode=mode,
        seed=seed_alg,
        verbose=True,
    )

    E_final = H_quad(G, m_final) / n

    print("\nHessian final result:")
    print("m_final =", m_final)
    print("||m_final||^2 / N =", (m_final @ m_final) / n)
    print("H(m_final)/N =", E_final)
    print("ratio to spectral =", E_final / E_spec)

    # Plot energy
    plot_energy(hist, E_spec=E_spec)

    # Plot trajectory
    if n == 2:
        plot_trajectory_2d(
            hist,
            n,
            m_spec,
            title="2D Hessian trajectory with spectral solution",
        )
    elif n == 3:
        plot_trajectory_3d(
            hist,
            n,
            m_spec,
            title="3D Hessian trajectory with spectral solution",
        )
    else:
        raise ValueError("This plotting script only supports n=2 or n=3.")