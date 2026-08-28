from mppt.jax_configs import x64; x64()
import numpy as np
from scipy.stats import norm
from mppt.abstract.pt_concept import geometric_anneal
from mppt.tests.pt_test_helper import dist_pt_intermediate_1d


def _test_pt_intermediate_1d(mean0, sigma0, mean1, sigma1, t, verbose):
    # The interpolating normal is only defined up to a normalizing constant in
    # log space, so fixed evaluation points give an exact shape/constant oracle.
    x = np.linspace(-2., 2., num=10)
    res = norm.logpdf(
        x, **dist_pt_intermediate_1d(mean0, sigma0, mean1, sigma1, t)._asdict())
    ref = geometric_anneal(t,
                           norm.logpdf(x, loc=mean0, scale=sigma0),
                           norm.logpdf(x, loc=mean1, scale=sigma1))
    delta = res - ref
    if verbose: print(delta)
    assert delta.shape == (10, )
    assert np.allclose(delta - delta[0], 0, atol=1e-10, rtol=1e-10)

def test_dist_pt_intermediate_1d():
    _test_pt_intermediate_1d(0.2, 3.4, 1.6, 2.15, 0.4, False)
    _test_pt_intermediate_1d(-0.7, 0.9, 1.2, 1.4, 0.63, False)
