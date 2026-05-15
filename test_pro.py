import numpy as np
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
import matplotlib.pyplot as plt


def GOE_generator(N, seed=None):
    """
    Generate an N x N GOE random matrix.

    Off-diagonal: A_ij ~ N(0, 1/N)
    Diagonal:     A_ii ~ N(0, 2/N)
    """
    rng = np.random.default_rng(seed)

    A = rng.normal(0, 1 / np.sqrt(N), size=(N, N))
    A = np.triu(A, 1)
    A = A + A.T

    diag = rng.normal(0, np.sqrt(2 / N), size=N)
    np.fill_diagonal(A, diag)

    return A


def solve_ising_gurobi(N, A, time_limit=60, verbose=False):
    """
    Solve

        max sigma^T A sigma
        s.t. sigma_i in {-1, +1}

    This is a nonconvex MIQP.
    """
    A = 0.5 * (A + A.T)

    model = gp.Model("ising_miqp")

    if not verbose:
        model.Params.OutputFlag = 0

    z = model.addMVar(N, vtype=GRB.BINARY, name="z")

    # Convert z in {0,1} to spin sigma in {-1,+1}
    sigma = 2 * z - 1

    obj = sigma @ A @ sigma

    model.setObjective(obj, GRB.MAXIMIZE)

    model.Params.NonConvex = 2

    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    model.optimize()

    if model.SolCount > 0:
        z_val = z.X
        sigma_val = 2 * z_val - 1
        return model.ObjVal, sigma_val, model.MIPGap
    else:
        return np.nan, None, np.nan


# def spectral_relaxation(A):
#     """
#     Solve the spherical relaxation exactly:
#
#         max sigma^T A sigma
#         s.t. ||sigma||_2 = sqrt(N)
#
#     The optimal value is
#
#         N * lambda_max(A)
#
#     Therefore value / N = lambda_max(A).
#     """
#     A = 0.5 * (A + A.T)
#     N = A.shape[0]
#
#     eigvals = np.linalg.eigvalsh(A)
#     lambda_max = eigvals[-1]
#
#     value = N * lambda_max
#
#     return value, lambda_max

def spectral_relaxation(A):
    """
    Spectral relaxation + sign rounding.

    Step 1:
        Solve continuous/spherical relaxation:
            max x^T A x
            s.t. ||x||_2 = sqrt(N)

    Step 2:
        Let v_max be the top eigenvector of A.
        Round it to Ising spins:
            sigma_i = sign(v_i)

    Returns:
        relaxed_value:  N * lambda_max(A)
        rounded_value:  sigma^T A sigma
        sigma:          rounded {-1, +1} vector
        lambda_max:     largest eigenvalue
    """
    A = 0.5 * (A + A.T)
    N = A.shape[0]

    eigvals, eigvecs = np.linalg.eigh(A)

    lambda_max = eigvals[-1]
    v_max = eigvecs[:, -1]

    # continuous relaxation value
    relaxed_value = N * lambda_max

    # sign rounding
    sigma = np.sign(v_max)
    sigma[sigma == 0] = 1

    # rounded Ising objective
    rounded_value = sigma @ A @ sigma

    return relaxed_value, rounded_value, sigma, lambda_max


def spectral_relaxation_v2(A):
    """
    Solve the spherical relaxation exactly:

        max sigma^T A sigma
        s.t. ||sigma||_2 = sqrt(N)

    The optimal value is

        N * lambda_max(A)

    Therefore value / N = lambda_max(A).
    """
    A = 0.5 * (A + A.T)
    N = A.shape[0]

    m = gp.Model("Convex_relax")
    sigma = m.addMVar((N), lb=-GRB.INFINITY, name='sigma')
    m.addConstr(sigma @ sigma == N, "sigma")
    obj = sigma @ A @ sigma

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    return m.ObjVal, sigma.X

def SDP_relaxation(N, A, solver="SCS"):
    """
    Solve the SDP relaxation:

        max <A, X>
        s.t. diag(X) = 1
             X >= 0

    This gives an upper bound of the Ising optimum.
    """
    A = 0.5 * (A + A.T)

    X = cp.Variable((N, N), symmetric=True)

    constraints = [
        cp.diag(X) == np.ones(N),
        X >> 0
    ]

    objective = cp.Maximize(cp.sum(cp.multiply(A, X)))

    problem = cp.Problem(objective, constraints)

    problem.solve(
        solver=solver,
        verbose=False,
        eps=1e-4,
        max_iters=5000
    )



    return problem.value, X.value


def run_probability_experiment(
    N=40,
    num_trials=20,
    time_limit=60,
    compute_sdp=True,
    compute_spectral_relaxation_v2=True,
    seed0=1000
):
    """
    Repeatedly sample A ~ GOE(N), then compute:

        Ising optimum / N
        Spectral relaxation / N
        SDP relaxation / N

    For larger N, exact Ising MIQP can be slow.
    """
    results = {
        "ising": [],
        "spectral": [],
        "spectral_rounded": [],
        "spectral_v2": [],
        "sdp": [],
        "mip_gap": []
    }

    for trial in range(num_trials):
        seed = seed0 + trial
        A = GOE_generator(N, seed=seed)

        print(f"\nTrial {trial + 1}/{num_trials}, seed={seed}")

        # Exact or near-exact Ising by Gurobi
        ising_value, sigma, mip_gap = solve_ising_gurobi(
            N,
            A,
            time_limit=time_limit,
            verbose=False
        )

        # Spherical / spectral relaxation
        # spectral_value, lambda_max = spectral_relaxation(A)
        spectral_value, spectral_rounded_value, sigma, lambda_max = spectral_relaxation(A)

        # SDP relaxation
        if compute_sdp:
            try:
                sdp_value, X_opt = SDP_relaxation(N, A, solver="SCS")
            except Exception as e:
                print("SDP failed:", e)
                sdp_value = np.nan
        else:
            sdp_value = np.nan


        # # compute_spectral_relaxation_v2 relaxation
        # if compute_spectral_relaxation_v2:
        #     spectral_value_v2, spectral_sigma = spectral_relaxation_v2(A)


        results["ising"].append(ising_value / N)
        results["spectral"].append(spectral_value / N)
        results["spectral_rounded"].append(spectral_rounded_value / N)
        # results["spectral_v2"].append(spectral_value_v2 / N)
        results["sdp"].append(sdp_value / N)
        results["mip_gap"].append(mip_gap)

        print(f"Ising optimum / N:     {ising_value / N:.4f}")
        print(f"Spectral value / N:    {spectral_value / N:.4f}")
        print(f"Spectral_rounded value / N:    {spectral_rounded_value / N:.4f}")
        # print(f"Spectral_v2 value / N:    {spectral_value_v2 / N:.4f}")
        # print(f"Spectral_v2 sigma:    {spectral_sigma}")
        print(f"SDP value / N:         {sdp_value / N:.4f}")
        print(f"Gurobi MIP gap:        {mip_gap:.4e}")

    return results


def summarize_results(results):
    """
    Print mean and standard deviation.
    """
    print("\n================ Summary ================")

    for key in ["ising", "spectral", "spectral_rounded", "sdp"]:
        arr = np.array(results[key], dtype=float)
        arr = arr[~np.isnan(arr)]

        if len(arr) == 0:
            continue

        print(f"{key:10s}: mean = {np.mean(arr):.4f}, std = {np.std(arr):.4f}")

    print("\nTheoretical asymptotic references:")
    print("Ising optimum / N  ≈ 1.526")
    print("Spectral value / N ≈ 2.000")
    print("SDP value / N      ≈ 2.000")


def plot_results(results):
    """
    Plot trial values.
    """
    plt.figure(figsize=(8, 5))

    for key in ["Ising", "Sectral", "SDP"]:
        arr = np.array(results[key], dtype=float)
        plt.plot(arr, marker="o", label=key)

    plt.axhline(1.526, linestyle="--", label="Ising asymptotic ≈ 1.526")
    plt.axhline(2.0, linestyle="--", label="Spectral/SDP asymptotic ≈ 2")

    plt.xlabel("Trial")
    plt.ylabel("Value / N")
    plt.title("GOE random instances")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    # For exact Gurobi MIQP, start with N=30~50.
    # Larger N may become slow.
    N = 200
    num_trials = 10

    results = run_probability_experiment(
        N=N,
        num_trials=num_trials,
        time_limit=60,
        compute_sdp=True,
        compute_spectral_relaxation_v2=True,
        seed0=42
    )

    summarize_results(results)
    plot_results(results)