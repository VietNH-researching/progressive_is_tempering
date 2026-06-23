r"""Provides visual and formal routines to verify that two samples are drawn
from the same distribution.
"""
from itertools import zip_longest

import numpy as np
from typing import Tuple, Union
from scipy.stats import random_correlation, rv_discrete
from functools import cached_property
from statsmodels.nonparametric.kde import KDEUnivariate
from matplotlib import pyplot as plt
from jax import numpy as jnp, Array

BIG_SAMPLE_SIZE = 50_000
BIG_GRID_SIZE = 2000
plt.style.use('ggplot')

def np_random_mean_and_cov(d: int, range_mean: Tuple[float, float] = (-1, 1),
                           range_std: Tuple[float, float] = (1, 2), seed=None) \
        -> Tuple[np.ndarray, np.ndarray]:
    r"""Generate a random vector in `R^d` and a random `dxd` matrix to be used as the
    mean and the covariance matrix of a multivariate gaussian distribution.
    The correlation matrix is generated so that their eigenvalues are uniformly distributed
    in the simplex `{x_1 + ... + x_d = 1; x_i >=0}`.

    :param d: dimension
    :param range_mean: range of the uniform distribution on which the mean is drawed
    :param range_std: range of the uniform distribution on which the standard deviation is drawed
    :return: mean and cov
    """
    if seed is not None:
        np.random.seed(seed)
    eigs = np.diff(sorted(np.r_[0, np.random.rand(d - 1), 1])) * d
    corr = random_correlation.rvs(eigs)
    mu = np.random.uniform(range_mean[0], range_mean[1], size=d)
    S = np.diag(np.random.uniform(range_std[0], range_std[1], size=d))
    return mu, S @ corr @ S.T

class FastKDE:
    def __init__(self, sample):
        self.sample = sample
        self.kde = KDEUnivariate(self.sample)
        self.kde.fit()

    @cached_property
    def support(self):
        return self.kde.support

    @cached_property
    def density(self):
        return self.kde.density

    def _evaluate_inside(self, x):
        idx = np.searchsorted(self.support, x)
        idx = np.minimum(idx, len(self.support) - 1)
        return self.density[idx]

    @cached_property
    def min_support(self):
        return np.min(self.support)

    @cached_property
    def max_support(self):
        return np.max(self.support)

    def evaluate(self, x):
        check_inside = np.logical_and(x >= self.min_support, x <= self.max_support)
        return np.where(check_inside, self._evaluate_inside(x), 0.)

def plot_continuous_samples(sample1, sample2, sample2_is_dist: bool) -> plt.Figure:
    r"""Plot the kernel density estimators of sample1 and sample2
    :param sample1: a sample of floats
    :param sample2: a sample of floats if sample2_is_dist is False.
    If sample2_is_dist is True, sample2 is a distribution-like object with a .rvs() method
    """
    kde1, kde2 = _get_kde(sample1, sample2, sample2_is_dist)
    return _plot_continuous(kde1=kde1, kde2=kde2)

def _plot_continuous(kde1, kde2):
    fig, ax = plt.subplots()
    ax.plot(kde1.support, kde1.density, label='1 (emp)')
    ax.plot(kde2.support, kde2.density, label='2 (true)')
    ax.legend()
    return fig

def _get_kde(sample1, sample2, sample2_is_dist):
    if sample2_is_dist:
        sample2 = sample2.rvs(size=BIG_SAMPLE_SIZE)
        assert len(sample2.shape) == 1
    kde1 = FastKDE(sample1)
    kde2 = FastKDE(sample2)
    return kde1, kde2

def TV_continuous_samples(sample1, sample2, sample2_is_dist: bool) -> float:
    r"""Estimate the total variation distance between two continuous distributions
    using two samples. The second sample can optionally be a probability distribution
    with a .rvs() method.
    """
    kde1, kde2 = _get_kde(sample1, sample2, sample2_is_dist)
    return _TV_continuous(kde1=kde1, kde2=kde2)

def _TV_continuous(kde1, kde2):
    xlow = min(min(kde1.support), min(kde2.support))
    xhigh = max(max(kde1.support), max(kde2.support))
    x = np.linspace(xlow, xhigh, BIG_GRID_SIZE)
    f = 0.5 * np.abs(kde2.evaluate(x) - kde1.evaluate(x))
    return (xhigh - xlow) * float(np.mean(f))

def plot_multivariate_samples(sample1, sample2, sample2_is_dist, projector):
    if sample2_is_dist:
        sample2 = sample2.rvs(size=BIG_SAMPLE_SIZE)
    dim = sample1.shape[1]
    assert dim == sample2.shape[1]
    return plot_continuous_samples(sample1 @ projector, sample2 @ projector, False)

def multivariate_sample_tester(sample1, sample2, sample2_is_dist,
                               TV_threshold, verbose, title=None):
    r"""Test whether a multivariate sample follows the prescribed distribution
    by projecting the sample over three random axes.
    """
    np.random.seed()
    for i in range(3):
        projector = np_uniform_vector_sphere(sample1.shape[1])
        tv = TV_multivariate_samples(sample1, sample2, sample2_is_dist, projector)
        if verbose:
            print('TV{}:'.format(i), tv)
        if verbose:
            fig = plot_multivariate_samples(sample1, sample2, sample2_is_dist, projector)
            fig.axes[0].set_title(title) if title is not None else None
            fig.show()
        assert tv < TV_threshold

def np_uniform_vector_sphere(dim):
    x = np.random.normal(size=dim)
    return x / np.linalg.norm(x)

def TV_multivariate_samples(sample1, sample2, sample2_is_dist, projector) -> float:
    if sample2_is_dist:
        sample2 = sample2.rvs(size=BIG_SAMPLE_SIZE)
    dim = sample1.shape[1]
    assert dim == sample2.shape[1]
    return TV_continuous_samples(sample1 @ projector, sample2 @ projector, False)

def discrete_pmf(x):
    r"""
    Given an array of `x` elements in {0, 1, ..., N-1}, returns
    the pmf vector of length N
    """
    if not isinstance(x, Array):
        x = jnp.array(x)
    values, counts = jnp.unique_counts(x)
    res = jnp.zeros(max(values) + 1, dtype=int)
    res = res.at[values].set(counts)
    assert jnp.allclose(counts.sum(), len(x))
    assert jnp.allclose(res.sum(), len(x))
    return res / len(x)

def TV_discrete_sample_unbounded(sample, dist: rv_discrete) -> Union[float, Array]:
    r"""Calculates the total variation distance between a sample and a discrete
    random variable in scipy which is supposed to be unbounded

    Warning: the sample is supposed to have nonnegative values only.
    """
    sample_pmf = discrete_pmf(sample)
    support = jnp.arange(jnp.max(sample) + 1)
    true_pmf = dist.pmf(support)
    return jnp.sum(jnp.abs(sample_pmf - true_pmf)) * 0.5

def plot_discrete_sample_unbounded(sample, dist):
    sample_pmf = discrete_pmf(sample)
    support = jnp.arange(jnp.max(sample) + 1)
    true_pmf = dist.pmf(support)
    fig, ax = plt.subplots()
    ax.scatter(support, sample_pmf, color='blue', label='emp')
    ax.scatter(support, true_pmf, color='green', label='true')
    ax.legend()
    fig.show()

def _discrete_pmf(x, flag):
    return x if flag else discrete_pmf(x)

def plot_discrete_sample_bounded(sample1, sample2, sample2_is_prob: bool) -> plt.Figure:
    r"""Plot the histograms of the discrete distributions of sample1 and sample2
    :param sample1: a sample of nonnegative integers from 0 to N
    :param sample2: same format as sample1 if sample2_is_proba is False.
    If sample2_is_proba is True, sample2 is the vector of probabilities that sample1
    is expected to follow.
    """
    pmf1 = _discrete_pmf(sample1, False)
    pmf2 = _discrete_pmf(sample2, sample2_is_prob)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(pmf1)), pmf1, label='1 (emp)')
    ax.scatter(np.arange(len(pmf2)), pmf2, label='2 (true)')
    ax.legend()
    return fig

def TV_discrete_samples_bounded(sample1, sample2, sample2_is_prob: bool) -> float:
    r"""Estimate the total variation distance between two discrete distributions
    using two samples. The second sample can optionally be a probability vector.
    """
    pmf1 = _discrete_pmf(sample1, False)
    pmf2 = _discrete_pmf(sample2, sample2_is_prob)
    res = 0.
    for p1, p2 in zip_longest(pmf1, pmf2, fillvalue=0.):
        res = res + 0.5 * np.abs(p1-p2)
    return res
