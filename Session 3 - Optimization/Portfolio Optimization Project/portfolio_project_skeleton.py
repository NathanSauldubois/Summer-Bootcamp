"""Student template for the Session 3 Portfolio Optimization Project.

Complete the TODO blocks in order. The function boundaries mirror the stages in
the project statement. Keep the validation checks: they are part of the work.

Suggested environment:
    python -m pip install numpy pandas scipy matplotlib

Run:
    python portfolio_project_skeleton.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing project dependency. Run: "
        "python -m pip install numpy pandas scipy matplotlib"
    ) from exc


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    n_assets: int = 10
    n_factors: int = 3
    n_train: int = 756
    n_test: int = 504
    annualization: int = 252
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001
    rebalance_every: int = 21
    estimation_window: int = 252
    weight_cap: float = 0.30
    output_dir: Path = Path("outputs")
    data_dir: Path = Path("data")


CFG = Config()


def make_output_directories(cfg: Config = CFG) -> None:
    """Create data, figure, and table folders without deleting existing work.

    Expected result: every configured path exists and is a directory. This
    function returns None and must be safe to call more than once.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)


def asset_names(n_assets: int) -> list[str]:
    """Return ordered, unique names Asset_01, ..., Asset_NN.

    TODO UTILITY: reject noninteger or nonpositive n_assets.
    """
    return [f"Asset_{j:02d}" for j in range(1, n_assets + 1)]


def check_covariance(sigma: np.ndarray, name: str, tol: float = 1e-10) -> None:
    """Fail early when a covariance matrix is malformed.

    The ValueError message must contain name, the failed property, and the
    observed shape or minimum eigenvalue so the problem can be located.
    """
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"{name}: covariance must be square, got {sigma.shape}")
    if not np.all(np.isfinite(sigma)):
        raise ValueError(f"{name}: non-finite entries")
    if not np.allclose(sigma, sigma.T, atol=tol, rtol=0.0):
        raise ValueError(f"{name}: covariance is not symmetric")
    smallest = float(np.linalg.eigvalsh(sigma).min())
    if smallest < -tol:
        raise ValueError(f"{name}: minimum eigenvalue is {smallest:.3e}")


# ---------------------------------------------------------------------------
# Stage 1: data generation
# ---------------------------------------------------------------------------

def factor_covariance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return factor means, volatilities, and annual covariance.

    TODO 1.1
    1. Create mu_f, factor_vols, and the correlation matrix from the statement.
    2. Compute sigma_f = diag(vols) @ corr @ diag(vols).
    3. Call check_covariance before returning.
    """
    raise NotImplementedError("TODO 1.1: build the factor covariance")


def generate_population_parameters(
    rng: np.random.Generator, cfg: Config = CFG
) -> dict[str, np.ndarray]:
    """Draw alpha, factor loadings B, and idiosyncratic covariance D.

    TODO 1.2
    Expected shapes:
        alpha: (n_assets,)
        B: (n_assets, n_factors)
        D: (n_assets, n_assets)

    Implementation recipe:
    - alpha = rng.uniform(-0.01, 0.03, size=n_assets)
    - B = rng.normal(0.0, 0.45, size=(n_assets, n_factors))
    - idio_vol = rng.uniform(0.10, 0.22, size=n_assets)
    - D is the diagonal matrix of idio_vol ** 2
    - call factor_covariance() for mu_f and sigma_f

    Return keys: alpha, B, D, idio_vol, mu_f, factor_vol, sigma_f.
    Assert every expected shape before returning.
    """
    raise NotImplementedError("TODO 1.2: generate population parameters")


def theoretical_moments(
    params: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute annual population mean and covariance.

    TODO 1.3
        mu_true = alpha + B @ mu_f
        sigma_true = B @ sigma_f @ B.T + D

    Check dimensions and covariance validity before returning.
    """
    raise NotImplementedError("TODO 1.3: compute theoretical moments")


def simulate_returns(
    rng: np.random.Generator,
    n_obs: int,
    params: dict[str, np.ndarray],
    cfg: Config = CFG,
) -> np.ndarray:
    """Simulate daily simple returns from the factor model.

    TODO 1.4
    Draw factors with
        rng.multivariate_normal(mu_f / 252, sigma_f / 252, size=n_obs)
    and idiosyncratic shocks with
        rng.multivariate_normal(zeros(n_assets), D / 252, size=n_obs).
    Combine them as
        alpha[None, :] / 252 + factors @ B.T + epsilon.

    Return an array with shape (n_obs, n_assets).
    """
    raise NotImplementedError("TODO 1.4: simulate one return sample")


def returns_frame(
    values: np.ndarray, start_date: str, cfg: Config = CFG
) -> pd.DataFrame:
    """Attach a business-day index and standard asset names."""
    dates = pd.bdate_range(start=start_date, periods=len(values))
    return pd.DataFrame(values, index=dates, columns=asset_names(cfg.n_assets))


def generate_and_save_data(cfg: Config = CFG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate independent train/test samples with shared parameters.

    TODO 1.5--1.7
    Use independent RNG streams. One simple approach is
        seed_sequence = np.random.SeedSequence(cfg.seed)
        rng_params, rng_train, rng_test = [
            np.random.default_rng(s) for s in seed_sequence.spawn(3)
        ]

    Save:
        data/returns_train.csv
        data/returns_test.csv
        data/true_parameters.npz

    Use non-overlapping date ranges and save the date index with label "Date".
    A simple choice is to start train at 2020-01-02 and test at the first
    business day after the final train date. Call theoretical_moments(), save
    mu_true and sigma_true with the primitive parameters, reload the files,
    and assert their expected shapes before returning.
    """
    raise NotImplementedError("TODO 1.5--1.7: generate and save all data")


def load_project_data(
    cfg: Config = CFG,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Load and validate the files produced in Stage 1.

    TODO 1.8
    1. Read both CSV files with index_col="Date" and parse_dates=True.
    2. Load the npz archive and convert it to a normal dictionary.
    3. Check row counts, exact asset names, increasing unique dates, finite
       values, and absence of overlap between train and test indices.
    4. Check that every stored population array has the documented shape.
    5. Return train, test, and parameters in that order.
    """
    raise NotImplementedError("TODO 1.8: load and validate project data")


def annualized_sample_moments(
    returns: pd.DataFrame, cfg: Config = CFG
) -> tuple[np.ndarray, np.ndarray]:
    """Compute annualized arithmetic mean and sample covariance."""
    mu_hat = cfg.annualization * returns.mean().to_numpy()
    sigma_hat = cfg.annualization * returns.cov().to_numpy()
    check_covariance(sigma_hat, "annualized sample covariance")
    return mu_hat, sigma_hat


# ---------------------------------------------------------------------------
# Stage 2: PCA
# ---------------------------------------------------------------------------

def fit_pca_on_train(
    train: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Fit PCA on centered training returns only.

    TODO 2.1--2.2
    Return a dictionary containing:
        mean: training column mean, shape (n_assets,)
        eigenvalues: decreasing daily covariance eigenvalues
        eigenvectors: matching columns
        explained_ratio: eigenvalues / eigenvalues.sum()
        empirical_covariance: daily sample covariance

    Implementation recipe:
    - training_mean = train.mean().to_numpy()
    - centered = train.to_numpy() - training_mean
    - empirical_covariance = centered.T @ centered / (n_obs - 1)
    - eigenvalues, eigenvectors = np.linalg.eigh(empirical_covariance)
    - reverse both outputs so eigenvalues are decreasing
    - explained_ratio = eigenvalues / eigenvalues.sum()

    Never center test data with the test mean. Verify U.T @ U is close to I
    and the full eigendecomposition reconstructs the empirical covariance.
    """
    raise NotImplementedError("TODO 2.1--2.2: fit PCA on training data")


def pca_covariance(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_components: int,
    empirical_covariance: np.ndarray,
    diagonal_correction: bool = True,
) -> np.ndarray:
    """Build a PCA covariance estimate.

    TODO 2.3--2.4
    First construct U_k @ diag(lambda_k) @ U_k.T.
    If diagonal_correction is True, add the diagonal of the residual
    empirical_covariance - low_rank_covariance.

    Symmetrize the numerical result with (S + S.T) / 2 and validate it.
    """
    raise NotImplementedError("TODO 2.3--2.4: reconstruct PCA covariance")


def pca_scores(
    returns: pd.DataFrame,
    training_mean: np.ndarray,
    training_eigenvectors: np.ndarray,
    n_components: int,
) -> pd.DataFrame:
    """Project returns with PCA quantities fitted on training data.

    TODO 2.5
    1. Verify column count and n_components.
    2. Center with training_mean, including when returns is the test sample.
    3. Compute scores = centered_values @ U_k.
    4. Return a DataFrame with the original dates and columns PC_01, ...
    """
    raise NotImplementedError("TODO 2.5: transform returns into PCA scores")


def principal_angles(
    empirical_vectors: np.ndarray,
    theoretical_vectors: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Return principal angles in radians between two leading subspaces.

    TODO 2.6
    Compute singular values of U_true.T @ U_empirical, clip to [-1, 1],
    and return arccos(singular_values).
    """
    raise NotImplementedError("TODO 2.6: compute principal angles")


def plot_pca_diagnostics(
    pca: dict[str, np.ndarray], output_path: Path
) -> None:
    """Create a two-panel scree/cumulative-explained-variance figure.

    TODO 2.7
    Label axes, mark the three-factor reference, call tight_layout(), and save.
    Close the figure after saving.
    """
    raise NotImplementedError("TODO 2.7: plot PCA diagnostics")


# ---------------------------------------------------------------------------
# Stage 3: covariance estimators and portfolio optimizers
# ---------------------------------------------------------------------------

def ridge_covariance(sigma: np.ndarray, tau: float) -> np.ndarray:
    """Return sigma + delta I with delta = tau * trace(sigma) / n."""
    # TODO 3.1: implement the formula and validate the result.
    raise NotImplementedError("TODO 3.1: ridge covariance")


def equal_weight(n_assets: int) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets)


def unconstrained_gmv(sigma: np.ndarray) -> np.ndarray:
    """Closed-form fully invested GMV without explicit inversion.

    TODO 3.2
    Solve sigma @ x = ones, then normalize x so its entries sum to one.
    """
    raise NotImplementedError("TODO 3.2: unconstrained GMV")


def solve_long_only_gmv(
    sigma: np.ndarray,
    weight_cap: float = 0.30,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, object]:
    """Solve bounded GMV with scipy.optimize.minimize.

    TODO 3.3
    Objective: 0.5 * w @ sigma @ w
    Jacobian: sigma @ w
    Equality: sum(w) - 1 = 0
    Bounds: (0, weight_cap) for every asset

    Use SLSQP, a feasible starting point, and strict tolerances, for example
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0.0, weight_cap)] * n_assets
    Check first that n_assets * weight_cap >= 1. Return both the weights and
    raw OptimizeResult so the report can show solver status.
    """
    raise NotImplementedError("TODO 3.3: long-only GMV")


def solve_max_sharpe(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float,
    weight_cap: float = 0.30,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, object]:
    """Solve long-only maximum Sharpe.

    TODO 3.4
    Minimize the negative Sharpe ratio subject to full investment and bounds.
    For one candidate w, compute excess = mu @ w - risk_free_rate and
    variance = w @ sigma @ w. Return a large penalty if variance is below a
    small tolerance. Try equal weight, long-only GMV, and several random
    feasible initializations in the calling code; keep only successful solves.
    """
    raise NotImplementedError("TODO 3.4: maximum-Sharpe portfolio")


def solve_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    gamma: float,
    weight_cap: float = 0.30,
) -> tuple[np.ndarray, object]:
    """Solve a bounded mean--variance problem.

    TODO 3.5
    Objective: 0.5 * w @ sigma @ w - gamma * mu @ w
    Gradient: sigma @ w - gamma * mu
    """
    raise NotImplementedError("TODO 3.5: mean--variance portfolio")


def trace_efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    target_returns: np.ndarray,
    weight_cap: float = 0.30,
) -> pd.DataFrame:
    """Solve one bounded minimum-variance problem per feasible target.

    TODO 3.6
    1. For each target m, minimize 0.5 * w @ sigma @ w.
    2. Impose sum(w) = 1 and mu @ w = m plus the weight bounds.
    3. Warm-start each solve with the preceding successful solution.
    4. Store target, realized predicted return, volatility, concentration,
       solver success, message, and maximum constraint violation.
    5. Do not discard failed targets: keep them in the returned table.
    """
    raise NotImplementedError("TODO 3.6: trace the efficient frontier")


def portfolio_diagnostics(
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    lower: float = 0.0,
    upper: float = 0.30,
) -> dict[str, float]:
    """Return feasibility, risk, return, and concentration diagnostics.

    TODO 3.7
    Include at least:
        budget_error, lower_violation, upper_violation,
        predicted_return, predicted_variance, predicted_volatility,
        concentration, min_weight, max_weight, finite_values.
    """
    raise NotImplementedError("TODO 3.7: portfolio diagnostics")


def directional_optimality_check(
    weights: np.ndarray,
    objective: Callable[[np.ndarray], float],
    lower: float = 0.0,
    upper: float = 0.30,
    n_directions: int = 200,
    step: float = 1e-6,
    seed: int = 123,
) -> dict[str, float]:
    """Search for small feasible perturbations that improve the objective.

    TODO 3.8
    1. Draw random directions and remove their component along the all-ones
       vector so each direction preserves the budget to first order.
    2. Scale or reject directions that violate the bounds.
    3. Evaluate objective(weights + step * direction) - objective(weights).
    4. Report the smallest change and the number of feasible directions tested.

    This is a numerical diagnostic, not a proof of optimality.
    """
    raise NotImplementedError("TODO 3.8: directional optimality check")


# ---------------------------------------------------------------------------
# Stage 4: chronological validation
# ---------------------------------------------------------------------------

def select_covariance_hyperparameters(
    train: pd.DataFrame,
    candidate_k: list[int],
    candidate_tau: list[float],
    validation_length: int = 126,
) -> pd.DataFrame:
    """Compare PCA and ridge choices without using test data.

    TODO 4.1--4.4
    Split train chronologically with iloc into estimation and validation
    blocks. Fit PCA only on the estimation block. For every candidate k and
    tau, estimate a GMV portfolio, hold it fixed over validation, and record
    annualized realized volatility plus concentration. Include baseline rows
    for empirical covariance and equal weight. Return one row per candidate
    with estimation_start, estimation_end, validation_start, validation_end,
    estimator, hyperparameter, solver_success, realized_volatility,
    concentration, and selection_rank. Choose parameters outside this function
    using a stated sorting rule and deterministic tie-breaking.
    """
    raise NotImplementedError("TODO 4.1--4.4: chronological validation")


# ---------------------------------------------------------------------------
# Stage 5: backtesting
# ---------------------------------------------------------------------------

def drift_weights(
    post_trade_weights: np.ndarray, realized_asset_return: np.ndarray
) -> np.ndarray:
    """Update weights after one observed return vector.

    TODO 5.1
    Use the drift formula from the statement. Check that the portfolio gross
    return denominator is positive and renormalize only for tiny roundoff.
    """
    raise NotImplementedError("TODO 5.1: weight drift")


def performance_metrics(
    net_returns: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    risk_free_rate: float = CFG.risk_free_rate,
    annualization: int = CFG.annualization,
) -> dict[str, float]:
    """Compute annualized return, volatility, Sharpe, MDD, and trading metrics.

    TODO 5.2
    Implementation recipe:
    - wealth = (1 + net_returns).cumprod()
    - annual return = wealth.iloc[-1] ** (252 / len(net_returns)) - 1
    - annual volatility = sqrt(252) * net_returns.std(ddof=1)
    - daily rf = (1 + risk_free_rate) ** (1 / 252) - 1
    - Sharpe = sqrt(252) * mean(net_returns - daily_rf) / daily_std
    - drawdown = 1 - wealth / wealth.cummax(); MDD = drawdown.max()
    - total cost = costs.sum()

    Average turnover is computed on rebalance dates; document how zero-trade
    dates are identified. Reject empty returns and non-finite wealth paths.
    """
    raise NotImplementedError("TODO 5.2: performance metrics")


def frozen_weight_backtest(
    test: pd.DataFrame,
    weights: np.ndarray,
    charge_initial_trade: bool = False,
    cfg: Config = CFG,
) -> dict[str, object]:
    """Evaluate one fixed target portfolio over the test sample.

    TODO 5.3
    1. Validate the initial weights and choose the stated initial-cost rule.
    2. Apply the fixed target weights to each daily asset-return vector.
       For this baseline, do not re-estimate moments during test.
    3. Store gross returns, costs, net returns, wealth, and drawdown with the
       same test index.
    4. Call performance_metrics and return both paths and summary metrics.

    State whether "fixed" means constant portfolio weights with implicit daily
    rebalancing or buy-and-hold drifting weights. Use the former here so the
    baseline has an unambiguous definition.
    """
    raise NotImplementedError("TODO 5.3: frozen-weight backtest")


def backtest_rebalanced_strategy(
    train: pd.DataFrame,
    test: pd.DataFrame,
    covariance_estimator: Callable[[pd.DataFrame], np.ndarray],
    optimizer: Callable[[np.ndarray], np.ndarray],
    cfg: Config = CFG,
    window: str = "rolling",
) -> dict[str, object]:
    """Backtest one covariance/optimizer pair with a strict one-period lag.

    TODO 5.4--5.8
    Suggested loop for each test position t:
    1. Build the information set using train and test rows strictly before t.
    2. If t is a rebalance date, estimate covariance and target weights.
    3. Compute turnover relative to current pre-trade weights.
    4. Apply post-trade weights to test.iloc[t].
    5. Deduct transaction cost.
    6. Drift the weights using the return just observed.
    7. Store target/pre-trade weights, gross/net return, turnover, and cost.

    Return a dictionary with target_weights, pre_trade_weights, gross_returns,
    net_returns, turnover, costs, wealth, drawdown, solver_log,
    estimation_windows, and metrics. Add an assertion that the first target
    weights do not depend on test.iloc[0].
    """
    raise NotImplementedError("TODO 5.4--5.8: rebalanced backtest")


def plot_backtest_diagnostics(
    backtests: dict[str, dict[str, object]], output_dir: Path
) -> None:
    """Create comparable wealth, drawdown, turnover, and weight figures.

    TODO 5.9
    Required output files:
        wealth_curves.png
        drawdown_curves.png
        turnover.png
        rolling_volatility.png
        weights_<strategy>.png for each rebalanced strategy

    Align dates before plotting, label every axis, use the same color per
    strategy across figures, and close every Matplotlib figure after saving.
    """
    raise NotImplementedError("TODO 5.9: backtest diagnostic plots")


# ---------------------------------------------------------------------------
# Stage 6: reporting and robustness
# ---------------------------------------------------------------------------

def run_one_seed(seed: int) -> pd.DataFrame:
    """Run the complete experiment for one seed and return strategy metrics.

    TODO 6.1--6.3
    This function should call the same pipeline used for the main seed. Do not
    duplicate logic. Hyperparameters must be selected again within that seed's
    training sample.
    """
    raise NotImplementedError("TODO 6.1--6.3: multi-seed experiment")


def run_sanity_checks() -> None:
    """Run small deterministic tests before launching the full experiment.

    TODO TESTS
    Include at least these checks:
    - equal_weight(n).sum() == 1;
    - unconstrained GMV on the identity covariance equals equal weight;
    - ridge_covariance increases every eigenvalue by the same delta;
    - PCA with all components reconstructs the sample covariance;
    - drift_weights preserves the budget;
    - zero asset returns leave weights unchanged;
    - a covariance with a negative eigenvalue is rejected;
    - changing test return at date t does not change the weight applied at t;
    - performance_metrics gives zero drawdown for a strictly increasing path.

    Use np.testing helpers and print one short success message only after all
    checks pass.
    """
    raise NotImplementedError("TODO TESTS: implement project sanity checks")


def main() -> None:
    make_output_directories()

    # Work stage by stage. Uncomment a block only after completing its TODOs.
    #
    # run_sanity_checks()
    # train, test = generate_and_save_data()
    # train, test, params = load_project_data()
    # pca = fit_pca_on_train(train)
    # train_scores = pca_scores(
    #     train, pca["mean"], pca["eigenvectors"], CFG.n_factors
    # )
    # test_scores = pca_scores(
    #     test, pca["mean"], pca["eigenvectors"], CFG.n_factors
    # )
    # plot_pca_diagnostics(
    #     pca, CFG.output_dir / "figures" / "pca_diagnostics.png"
    # )
    #
    # mu_hat, sigma_hat = annualized_sample_moments(train)
    # w_ew = equal_weight(CFG.n_assets)
    # w_gmv = unconstrained_gmv(sigma_hat)
    # print(portfolio_diagnostics(w_gmv, mu_hat, sigma_hat))
    #
    # selection = select_covariance_hyperparameters(
    #     train, candidate_k=[1, 2, 3, 5],
    #     candidate_tau=[1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    # )
    # selection.to_csv(
    #     CFG.output_dir / "tables" / "hyperparameter_selection.csv",
    #     index=False,
    # )
    #
    # Continue with frozen and rebalanced backtests, diagnostic plots, then
    # run_one_seed for the robustness experiment.
    print(
        "Template ready. Complete TODO blocks in project order, "
        "starting with generate_and_save_data()."
    )


if __name__ == "__main__":
    main()
