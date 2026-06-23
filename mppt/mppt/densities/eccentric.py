from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal as jmvn
from jax.scipy.special import logsumexp, softmax
from scipy.stats import norm as snorm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def jax_logpdf(x):
    """Unvectorized"""
    d, = x.shape
    v1 = jmvn.logpdf(x, mean=jnp.zeros(d), cov=1/16) + jnp.log(0.5)
    v2 = jmvn.logpdf(x, mean=jnp.zeros(d) + 3/4, cov=1/16) + jnp.log(0.5)
    return logsumexp(jnp.array([v1, v2]))

def numpy_logpdf(x):
    """Vectorized"""
    v1 = np.sum(snorm.logpdf(x, loc=0, scale=1/4), axis=1) + np.log(0.5)
    v2 = np.sum(snorm.logpdf(x, loc=3/4, scale=1/4), axis=1) + np.log(0.5)
    return np_logsumexp(v1, v2)

def np_logsumexp(x, y):
    m = np.minimum(x, y)
    M = np.maximum(x, y)
    return M + np.log1p(np.exp(m-M))

def validate_1d(x):
    r"""Test whether the obtained sample follows the target distribution. The test
    only operates on the first component of the sample.
    It divides the sample into two sets: the first one from N(0, 1/4) and the second
    one from N(3/4, 1/4). It then prints out the proportion between the two and
    a graph comparing each set with the true distribution, allowing the user to
    judge visually whether the obtained sample is reasonable.
    """
    assert x.ndim == 1
    w0 = snorm.logpdf(x, loc=0, scale=1/4)
    w1 = snorm.logpdf(x, loc=3/4, scale=1/4)
    prob_1 = softmax(jnp.array([w0, w1]), axis=0)[1]
    assert prob_1.shape == x.shape
    idx = np.random.uniform(size=prob_1.shape) < prob_1
    print('Proportion:', 1-np.mean(idx), '|', np.mean(idx))
    check_sample_gaussian(x[np.logical_not(idx)], loc=0, scale=1/4)
    check_sample_gaussian(x[idx], loc=3/4, scale=1/4)

def check_sample_gaussian(x, loc, scale):
    fig, ax = plt.subplots()
    xx = np.linspace(loc - 4 * scale, loc + 4 * scale, num=1000)
    yy = snorm.pdf(xx, loc=loc, scale=scale)
    ax.plot(xx, yy, label='true')
    sns.kdeplot(x, ax=ax, label='empirical')
    ax.legend()
    fig.show()