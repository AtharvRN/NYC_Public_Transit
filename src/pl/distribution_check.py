from scipy.stats import norm,poisson,nbinom,chi2
from math import lgamma
import numpy as np
import matplotlib.pyplot as plt


def chi_squared(observed: list, expected: list):
    l = len(observed)
    chi = 0.0
    for i in range(l):
        if expected[i] <= 0:
            # skip bins with zero expected count to avoid division by zero
            continue
        dummy = ((observed[i] - expected[i])**2 / expected[i])
        chi += dummy
    return chi

def pick_quantile(distribution: list):
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q = [p / 100 for p in percentiles]
    qs = np.quantile(distribution, q)
    return qs
def poisson_pmf(k:int, lam:float):
    log_p = -lam + k * np.log(lam) - lgamma(k + 1)
    return np.exp(log_p)



def poisson_check(emp_dist: np.array):
    dist_mean = np.mean(emp_dist)
    dist_var = np.var(emp_dist, ddof=1)
    dispersion = dist_var / dist_mean

    bin_n = 50
    obs_mean = 0
    while obs_mean < 5 and bin_n > 1:
        obs, bin_edges = np.histogram(emp_dist, bins=bin_n)
        values = bin_edges[:-1]
        observed = obs.tolist()
        obs_mean = np.mean(np.array(observed))
        bin_n -= 2
    print(f'Final Bin values: {bin_n+2}')

    n = emp_dist.shape[0]
    expected = []
    for val in values:
        expected.append(n * poisson_pmf(val, dist_mean))

    DoF = len(observed)-2
    chi_score = chi_squared(observed, expected)
    p_value = chi2.sf(chi_score, DoF)
    emp_quantile = pick_quantile(emp_dist)
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q = [p / 100 for p in percentiles]
    poisson_quantile = [poisson.ppf(i, dist_mean) for i in q]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution comparison plot
    x_range = np.arange(0, int(np.max(emp_dist)) + 1)
    poisson_pmf_vals = poisson.pmf(x_range, dist_mean)

    ax1.hist(emp_dist, bins=30, density=True, alpha=0.6, label='Empirical', color='blue')
    ax1.plot(x_range, poisson_pmf_vals, 'r-', lw=2, label='Poisson')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability/Density')
    ax1.set_title('Empirical vs Poisson Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QQ plot
    ax2.plot(emp_quantile, poisson_quantile, 'o', label='Q-Q points')
    # Add diagonal reference line
    min_val = min(min(emp_quantile), min(poisson_quantile))
    max_val = max(max(emp_quantile), max(poisson_quantile))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    ax2.set_xlabel('Empirical Quantiles')
    ax2.set_ylabel('Poisson Quantiles')
    ax2.set_title('Poisson Q-Q Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'chi_score': chi_score,
        'dispersion': dispersion,
        'emp_quantile': emp_quantile,
        'poisson_quantile': poisson_quantile,
        'p_value': p_value
    }
def nb_param(emp_dist:np.array):
    n = emp_dist.shape[0]
    dist_mean = np.mean(emp_dist)
    dist_var = np.var(emp_dist,ddof=1)
    dispersion = dist_var / dist_mean
    if dist_var <= dist_mean:
        raise ValueError('nb does not fit undispersion')
    k = dist_mean**2 / (dist_var - dist_mean)
    n_param = k
    p_param = n_param / (n_param + dist_mean)
    return n_param, p_param

def nb_check(emp_dist: np.array):

    n_param, p_param = nb_param(emp_dist)

    dist_mean = np.mean(emp_dist)
    dist_var = np.var(emp_dist, ddof=1)
    dispersion = dist_var / dist_mean

    bin_n = 50
    obs_mean = 0
    while obs_mean < 5 and bin_n > 1:
        obs, bin_edges = np.histogram(emp_dist, bins=bin_n)
        values = bin_edges[:-1]
        observed = obs.tolist()
        obs_mean = np.mean(np.array(observed))
        bin_n -= 2
    print(f"Final Bin values (NB): {bin_n+2}")

    n = emp_dist.shape[0]
    # Expected counts per bin under fitted NB
    expected = []
    for i in range(len(values)):
        low = values[i]
        high = bin_edges[i+1]
        lo_int = int(np.floor(low))
        hi_int = int(np.floor(high)) - 1

        if hi_int < lo_int:
            expected.append(0.0)
            continue

        cdf_low = nbinom.cdf(lo_int - 1, n_param, p_param) if lo_int > 0 else 0.0
        cdf_high = nbinom.cdf(hi_int, n_param, p_param)
        prob_bin = max(cdf_high - cdf_low, 0.0)

        expected.append(prob_bin * n)

    DoF = len(observed)-3
    chi_score = chi_squared(observed, expected)
    p_value = chi2.sf(chi_score, DoF)
    emp_quantile = pick_quantile(emp_dist.tolist())
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    q = [p / 100 for p in percentiles]
    nb_quantile = [nbinom.ppf(p, n_param, p_param) for p in q]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution comparison plot
    x_range = np.arange(0, int(np.max(emp_dist)) + 1)
    nb_pmf = nbinom.pmf(x_range, n_param, p_param)

    ax1.hist(emp_dist, bins=30, density=True, alpha=0.6, label='Empirical', color='blue')
    ax1.plot(x_range, nb_pmf, 'r-', lw=2, label='Negative Binomial')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability/Density')
    ax1.set_title('Empirical vs Negative Binomial Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QQ plot
    ax2.plot(emp_quantile, nb_quantile, 'o', label='Q-Q points')
    # Add diagonal reference line
    min_val = min(min(emp_quantile), min(nb_quantile))
    max_val = max(max(emp_quantile), max(nb_quantile))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    ax2.set_xlabel('Empirical Quantiles')
    ax2.set_ylabel('NB Quantiles')
    ax2.set_title('Negative Binomial Q-Q Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "chi_score": chi_score,
        "dispersion": dispersion,
        "emp_quantile": emp_quantile,
        "nb_quantile": nb_quantile,
        "n_param": n_param,
        "p_param": p_param,
        "p_value": p_value
    }


