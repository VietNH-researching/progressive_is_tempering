from collections import namedtuple
from particles.smc_samplers import \
    TemperingBridge, ArrayMetropolis, view_2d_array, AdaptiveMCMCSequence, AdaptiveTempering
from particles import distributions as dists, SMC
import numpy as np
from scipy import stats
from particles.resampling import systematic
from typing import Union, Callable

class ArrayRandomWalkFixedScale(ArrayMetropolis):
    """Gaussian random walk Metropolis with fixed scale."""
    def __init__(self, scale):
        super().__init__()
        self._mh_scale = scale

    def calibrate(self, W, x):
        pass

    def proposal(self, x, xprop):
        arr = view_2d_array(x.theta)
        arr_prop = view_2d_array(xprop.theta)
        arr_prop[:, :] = arr + stats.norm.rvs(size=arr.shape) * self._mh_scale
        return 0.0

SMCSamplersResult = namedtuple('SMCSamplersResult',
                               ['x', 'logZ', 'temperatures'])


def smc_samplers(
        v_logpdf: Callable[[np.ndarray], np.ndarray], init_dist: Union[int, dists.ProbDist],
        n_mcmc_steps: int, n_particles: int, scale=None, verbose=True,
        ESSrmin=0.5) -> SMCSamplersResult:
    r"""Sample from a target distribution using Nicolas Chopin's particles package
    :param v_logpdf: vectorized logpdf of the target
    :param init_dist: the initial distribution (a `particles.ProbDist` object). Otherwise
    an integer must be given and the init_dist is set to the standard normal in that
    dimension.
    :param n_mcmc_steps: no of MCMC steps in the SMC Sampler
    :param n_particles: no of particles
    :param scale: if not given, adapt the scale and the covariance matrix using 2.38 rule
    """
    if not isinstance(init_dist, dists.ProbDist):
        init_dist = dists.MvNormal(cov=np.identity(init_dist))

    class ToyBridge(TemperingBridge):
        @staticmethod
        def logtarget(theta):
            return v_logpdf(theta)

    toy_bridge = ToyBridge(base_dist=init_dist)

    if scale is None:
        move = None
    else:
        mcmc = ArrayRandomWalkFixedScale(scale=scale)
        move = AdaptiveMCMCSequence(mcmc, len_chain=n_mcmc_steps)

    fk = AdaptiveTempering(toy_bridge, wastefree=False, len_chain=n_mcmc_steps,
                           move=move, ESSrmin=ESSrmin)
    alg = SMC(fk=fk, N=n_particles, verbose=verbose)
    alg.run()

    X = alg.X.theta
    assert isinstance(X, np.ndarray)
    X = X[systematic(alg.W)]

    return SMCSamplersResult(X, alg.logLt, alg.X.shared["exponents"])