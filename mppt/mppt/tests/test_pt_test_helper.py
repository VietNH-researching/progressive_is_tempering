from mppt.jax_configs import x64; x64()
import numpy as np
from scipy.stats import norm
from mppt.abstract.pt_concept import geometric_anneal
from mppt.tests.pt_test_helper import dist_pt_intermediate_1d


def _test_pt_intermediate_1d(mean0, sigma0, mean1, sigma1, t, verbose):
    x = np.random.normal(size=10)
    res = norm.logpdf(
        x, **dist_pt_intermediate_1d(mean0, sigma0, mean1, sigma1, t)._asdict())
    ref = geometric_anneal(t,
                           norm.logpdf(x, loc=mean0, scale=sigma0),
                           norm.logpdf(x, loc=mean1, scale=sigma1))
    delta = res - ref
    if verbose: print(delta)
    assert delta.shape == (10, )
    assert np.allclose(delta - delta[0], 0)

def test_dist_pt_intermediate_1d():
    _test_pt_intermediate_1d(0.2, 3.4, 1.6, 2.15, 0.4, False)
    _test_pt_intermediate_1d(np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand(), False)
