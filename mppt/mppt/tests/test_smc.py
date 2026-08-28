from mppt.jax_configs import x64; x64()

import pytest

from mppt.abstract.smc import _resolve_logZ_estimator
from mppt.impl.logZ_estimators import robust


def test_smc_logZ_estimator_names_are_validated():
    r"""SMC estimator names should fail with a stable boundary error.

    Plausible agent error: typoing an estimator name and getting an opaque
    ``AttributeError`` while tracing a jitted SMC step.
    """
    assert _resolve_logZ_estimator('robust') is robust
    with pytest.raises(ValueError, match="Unknown SMC logZ estimator"):
        _resolve_logZ_estimator('bogus')
