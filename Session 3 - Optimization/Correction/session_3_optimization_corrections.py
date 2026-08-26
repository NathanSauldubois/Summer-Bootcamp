"""Reference computations for the Session 3 optimization exercise corrections.

This file covers Exercises 10 and 11 only.  It deliberately does not contain
a solution to the separate Portfolio Optimization Project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE / "assets"


def centered_gradient(f, x, h=1e-6):
    """Return the centered finite-difference gradient of ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    approximation = np.empty_like(x)
    for j in range(x.size):
        direction = np.zeros_like(x)
        direction[j] = h
        approximation[j] = (f(x + direction) - f(x - direction)) / (2.0 * h)
    return approximation


def constant_step(alpha):
    """Create a constant step-size rule with the common rule interface."""
    return lambda _f, _x, _g: float(alpha)


def armijo_step(f, x, g, initial=1.0, c=1e-4, contraction=0.5):
    """Backtrack along the negative gradient until Armijo's test holds."""
    alpha = float(initial)
    fx = f(x)
    squared_norm = g @ g
    while f(x - alpha * g) > fx - c * alpha * squared_norm:
        alpha *= contraction
        if alpha < np.finfo(float).eps:
            raise RuntimeError("Armijo backtracking reached machine precision")
    return alpha


def gradient_descent(f, grad, x0, step_rule, tol=1e-8, max_iter=25_000):
    """Run gradient descent and retain all diagnostics requested in Exercise 10."""
    x = np.asarray(x0, dtype=float).copy()
    iterates, values, gradient_norms, step_sizes = [], [], [], []
    status = "maximum iterations reached"

    for _ in range(max_iter + 1):
        value = float(f(x))
        g = np.asarray(grad(x), dtype=float)
        gradient_norm = float(np.linalg.norm(g))
        iterates.append(x.copy())
        values.append(value)
        gradient_norms.append(gradient_norm)

        if not np.isfinite(value) or not np.all(np.isfinite(g)):
            status = "non-finite iterate"
            break
        if gradient_norm <= tol:
            status = "converged"
            break

        alpha = float(step_rule(f, x, g))
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("the step-size rule must return a finite positive value")
        step_sizes.append(alpha)
        x = x - alpha * g

    return {
        "x": x,
        "iterates": np.asarray(iterates),
        "values": np.asarray(values),
        "gradient_norms": np.asarray(gradient_norms),
        "step_sizes": np.asarray(step_sizes),
        "iterations": len(step_sizes),
        "status": status,
    }


def make_quadratic_instance(seed=3, dimension=20):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    q = matrix.T @ matrix + 0.1 * np.eye(dimension)
    c = rng.normal(size=dimension)
    x_star = np.linalg.solve(q, c)
    return q, c, x_star


def run_exercise_10():
    q, c, x_star = make_quadratic_instance()
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    lambda_min, lambda_max = eigenvalues[[0, -1]]
    f_quad = lambda x: 0.5 * x @ q @ x - c @ x
    g_quad = lambda x: q @ x - c
    f_star = f_quad(x_star)

    rules = {
        r"$1/L$": constant_step(1.0 / lambda_max),
        r"$2/(\mu+L)$": constant_step(2.0 / (lambda_min + lambda_max)),
        r"$2.05/L$": constant_step(2.05 / lambda_max),
    }
    quadratic_runs = {
        name: gradient_descent(f_quad, g_quad, np.zeros(q.shape[0]), rule)
        for name, rule in rules.items()
    }

    rosenbrock = lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    def rosenbrock_gradient(x):
        return np.array(
            [
                -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] ** 2),
            ]
        )

    x0_rosenbrock = np.array([-1.2, 1.0])
    finite_difference_error = np.linalg.norm(
        centered_gradient(rosenbrock, x0_rosenbrock)
        - rosenbrock_gradient(x0_rosenbrock),
        ord=np.inf,
    )
    rosenbrock_run = gradient_descent(
        rosenbrock,
        rosenbrock_gradient,
        x0_rosenbrock,
        lambda f, x, g: armijo_step(f, x, g),
    )

    import matplotlib.pyplot as plt

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name, run in quadratic_runs.items():
        gap = np.maximum(run["values"] - f_star, np.finfo(float).tiny)
        axes[0].semilogy(gap, label=name)
        axes[1].semilogy(run["gradient_norms"], label=name)
    axes[0].set(title="Quadratic objective gap", xlabel="iteration", ylabel=r"$f(x_k)-f(x^*)$")
    axes[1].set(title="Quadratic gradient norm", xlabel="iteration", ylabel=r"$\|\nabla f(x_k)\|_2$")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "exercise10_quadratic_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    stable_run = quadratic_runs[r"$2/(\mu+L)$"]
    error = stable_run["iterates"] - x_star
    coordinates = error @ eigenvectors[:, [0, -1]]
    axes[0].plot(coordinates[:, 0], coordinates[:, 1], lw=1.0)
    axes[0].scatter(coordinates[0, 0], coordinates[0, 1], marker="x", label="start")
    axes[0].scatter(0.0, 0.0, marker="*", label="optimum")
    axes[0].set(
        title="Quadratic path in extreme eigendirections",
        xlabel=r"coordinate for $\lambda_{\min}$",
        ylabel=r"coordinate for $\lambda_{\max}$",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    grid_x = np.linspace(-2.0, 1.25, 400)
    grid_y = np.linspace(-0.5, 2.0, 400)
    xx, yy = np.meshgrid(grid_x, grid_y)
    zz = 100.0 * (yy - xx**2) ** 2 + (1.0 - xx) ** 2
    axes[1].contour(xx, yy, zz, levels=np.logspace(-1, 3.5, 16), cmap="viridis")
    path = rosenbrock_run["iterates"]
    # The full history is dense; subsampling keeps the path legible.
    draw = np.unique(np.r_[np.arange(min(150, len(path))), np.arange(150, len(path), 100)])
    axes[1].plot(path[draw, 0], path[draw, 1], color="tab:red", lw=1.0)
    axes[1].scatter(*path[0], marker="x", color="black", label="start")
    axes[1].scatter(1.0, 1.0, marker="*", color="gold", edgecolor="black", label="optimum")
    axes[1].set(title="Armijo path on Rosenbrock", xlabel="$x$", ylabel="$y$")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "exercise10_paths.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "condition_number": lambda_max / lambda_min,
        "quadratic_runs": quadratic_runs,
        "rosenbrock_run": rosenbrock_run,
        "finite_difference_error": finite_difference_error,
    }


def make_equality_instance(seed=11, dimension=20, constraints=3):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dimension, dimension))
    q = matrix.T @ matrix + 0.1 * np.eye(dimension)
    a = rng.normal(size=(constraints, dimension))
    c = rng.normal(size=dimension)
    x_reference = rng.normal(size=dimension)
    b = a @ x_reference
    kkt = np.block([[q, a.T], [a, np.zeros((constraints, constraints))]])
    solution = np.linalg.solve(kkt, np.r_[-c, b])
    return q, a, b, c, solution[:dimension], solution[dimension:]


def primal_dual(q, a, b, c, x_star, lambda_star, alpha, beta, rho=0.0, max_iter=20_000):
    """Simultaneous descent--ascent for the ordinary or augmented Lagrangian."""
    x = np.zeros_like(c)
    multiplier = np.zeros_like(b)
    histories = {name: [] for name in ("primal_error", "feasibility", "stationarity", "lagrangian")}
    status = "maximum iterations reached"

    for _ in range(max_iter + 1):
        feasibility = a @ x - b
        stationarity = q @ x + c + a.T @ multiplier
        histories["primal_error"].append(np.linalg.norm(x - x_star))
        histories["feasibility"].append(np.linalg.norm(feasibility))
        histories["stationarity"].append(np.linalg.norm(stationarity))
        histories["lagrangian"].append(0.5 * x @ q @ x + c @ x + multiplier @ feasibility)

        if max(histories["feasibility"][-1], histories["stationarity"][-1]) <= 1e-8:
            status = "converged"
            break
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e80:
            status = "diverged"
            break

        augmented_gradient = stationarity + rho * a.T @ feasibility
        # Both updates use residuals evaluated at (x_k, lambda_k).
        x = x - alpha * augmented_gradient
        multiplier = multiplier + beta * feasibility

    return {
        **{name: np.asarray(values) for name, values in histories.items()},
        "x": x,
        "lambda": multiplier,
        "iterations": len(histories["primal_error"]) - 1,
        "status": status,
        "multiplier_error": np.linalg.norm(multiplier - lambda_star),
    }


def iteration_spectral_radius(q, a, alpha, beta, rho=0.0):
    d, m = q.shape[0], a.shape[0]
    transition = np.block(
        [
            [np.eye(d) - alpha * (q + rho * a.T @ a), -alpha * a.T],
            [beta * a, np.eye(m)],
        ]
    )
    return float(np.max(np.abs(np.linalg.eigvals(transition))))


def run_exercise_11():
    q, a, b, c, x_star, lambda_star = make_equality_instance()
    configurations = {
        "stable (0.02, 0.02)": (0.02, 0.02),
        "oscillatory (0.03, 0.03)": (0.03, 0.03),
    }
    runs = {
        name: primal_dual(q, a, b, c, x_star, lambda_star, alpha, beta)
        for name, (alpha, beta) in configurations.items()
    }
    radii = {
        name: iteration_spectral_radius(q, a, alpha, beta)
        for name, (alpha, beta) in configurations.items()
    }
    augmented = {
        rho: primal_dual(q, a, b, c, x_star, lambda_star, 0.005, 0.02, rho=rho)
        for rho in (0.1, 1.0, 10.0)
    }
    augmented_radii = {
        rho: iteration_spectral_radius(q, a, 0.005, 0.02, rho=rho)
        for rho in augmented
    }

    import matplotlib.pyplot as plt

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name, run in runs.items():
        draw = slice(None) if run["status"] == "converged" else slice(0, min(250, len(run["feasibility"])))
        axes[0].semilogy(run["feasibility"][draw], label=name)
        axes[1].semilogy(run["stationarity"][draw], label=name)
    axes[0].set(title="Feasibility residual", xlabel="iteration", ylabel=r"$\|Ax_k-b\|_2$")
    axes[1].set(title="Stationarity residual", xlabel="iteration", ylabel=r"$\|Qx_k+c+A^T\lambda_k\|_2$")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "exercise11_primal_dual.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for rho, run in augmented.items():
        axes[0].semilogy(run["feasibility"], label=fr"$\rho={rho:g}$")
        axes[1].semilogy(run["stationarity"], label=fr"$\rho={rho:g}$")
    axes[0].set(title="Augmented: feasibility", xlabel="iteration", ylabel="residual")
    axes[1].set(title="Augmented: stationarity", xlabel="iteration", ylabel="residual")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "exercise11_augmented.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "runs": runs,
        "radii": radii,
        "augmented": augmented,
        "augmented_radii": augmented_radii,
        "x_star": x_star,
        "lambda_star": lambda_star,
    }


def main():
    ex10 = run_exercise_10()
    print("Exercise 10")
    print(
        f"lambda_min={ex10['lambda_min']:.9f}, "
        f"lambda_max={ex10['lambda_max']:.9f}, "
        f"condition number={ex10['condition_number']:.3f}"
    )
    for name, run in ex10["quadratic_runs"].items():
        print(
            f"  {name}: {run['status']}, iterations={run['iterations']}, "
            f"gradient norm={run['gradient_norms'][-1]:.3e}"
        )
    rosenbrock = ex10["rosenbrock_run"]
    print(
        f"  Rosenbrock: {rosenbrock['status']}, iterations={rosenbrock['iterations']}, "
        f"x={rosenbrock['x']}, gradient norm={rosenbrock['gradient_norms'][-1]:.3e}"
    )
    print(f"  finite-difference error={ex10['finite_difference_error']:.3e}")

    ex11 = run_exercise_11()
    print("Exercise 11")
    for name, run in ex11["runs"].items():
        print(
            f"  {name}: spectral radius={ex11['radii'][name]:.6f}, "
            f"{run['status']}, iterations={run['iterations']}"
        )
    for rho, run in ex11["augmented"].items():
        print(
            f"  rho={rho:g}: spectral radius={ex11['augmented_radii'][rho]:.6f}, "
            f"{run['status']}, iterations={run['iterations']}"
        )


if __name__ == "__main__":
    main()
