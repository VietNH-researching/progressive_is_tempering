from mppt.jax_configs import x64; x64()

import pytest

from mppt.ui import ui_main
from mppt.ui.collectors import parse_collectors
from mppt.ui.engine_ui import parse_engine
from mppt.ui.explorers import parse_explorer
from mppt.ui.swappers_api import parse_swappers


def test_pt_manual_runners_forwards_swap_counts(monkeypatch):
    r"""Explicit swap counts must keep their public meanings in manual mode.

    Regression test for accidentally passing n_inner_swaps and n_outer_swaps
    to the shared PT-step builder in the opposite order.
    """
    captured = {}

    def fake_build_pt_step(problem, temperatures, engine, explorer, collectors, *,
                           n_outer_swaps, n_inner_swaps, swapper):
        captured['n_outer_swaps'] = n_outer_swaps
        captured['n_inner_swaps'] = n_inner_swaps

        def step(carry, _, last_iter=False):
            return carry, {}

        return step, object()

    def fake_compile_all(compact_step, full_step, smc_init, cold_init):
        return compact_step, full_step, smc_init, cold_init

    monkeypatch.setattr(ui_main, '_build_pt_step', fake_build_pt_step)
    monkeypatch.setattr(ui_main, '_compile_all', fake_compile_all)

    ui_main.pt_manual_runners(
        problem=None,
        temperatures=[0., 0.5, 1.],
        n_particles=2,
        n_inner_swaps=7,
        n_outer_swaps=11,
    )

    assert captured == {'n_outer_swaps': 11, 'n_inner_swaps': 7}


def test_build_pt_step_requires_keyword_swap_counts():
    r"""Swap-count plumbing is deliberately keyword-only.

    ``n_inner_swaps`` and ``n_outer_swaps`` are both optional integers, so
    accepting them positionally makes silent transposition too easy.
    """
    with pytest.raises(TypeError):
        ui_main._build_pt_step(None, [0., 1.], {}, {}, [], 11, 7, {})

def test_public_runners_reject_invalid_control_counts_early():
    r"""Invalid loop/particle counts should fail before JAX tracing.

    Plausible agent error: passing ``0``/``None``/negative values into a runner
    and getting a late TypeError or silent no-op in ``lax.fori_loop``.
    """
    with pytest.raises(ValueError, match="n_iter"):
        ui_main.parallel_tempering(
            key=None, problem=None, temperatures=[0., 1.],
            n_iter=0, n_particles=2, verbose=False
        )

    with pytest.raises(ValueError, match="n_particles"):
        ui_main.parallel_tempering(
            key=None, problem=None, temperatures=[0., 1.],
            n_iter=1, n_particles=None, verbose=False
        )

    with pytest.raises(ValueError, match="n_particles"):
        ui_main.pt_manual_runners(
            problem=None, temperatures=[0., 1.], n_particles=0
        )

def test_build_pt_step_rejects_negative_swap_counts():
    r"""Negative swap counts are invalid even though JAX loops can no-op."""
    kwargs = dict(
        problem=None,
        temperatures=[0., 1.],
        engine={'name': 'random_walk'},
        explorer={'mode': 'adaptive'},
        collectors=[],
        swapper={'mix': True},
    )

    with pytest.raises(ValueError, match="n_outer_swaps"):
        ui_main._build_pt_step(
            **kwargs, n_outer_swaps=-1, n_inner_swaps=3
        )
    with pytest.raises(ValueError, match="n_inner_swaps"):
        ui_main._build_pt_step(
            **kwargs, n_outer_swaps=1, n_inner_swaps=-1
        )

def test_smc_sampler_rejects_invalid_controls_before_initialization():
    r"""SMC control parameters should fail before sampling or compilation."""
    base = dict(
        key=None, problem=None, n_particles=2,
        engine={'name': 'random_walk'}, n_mcmc_steps=1,
        local_ess_ratio=0.5, verbose=False, runtime_check=False,
    )

    with pytest.raises(ValueError, match="n_particles"):
        ui_main.smc_sampler(**{**base, 'n_particles': 0})
    with pytest.raises(ValueError, match="n_mcmc_steps"):
        ui_main.smc_sampler(**{**base, 'n_mcmc_steps': -1})
    with pytest.raises(ValueError, match="local_ess_ratio"):
        ui_main.smc_sampler(**{**base, 'local_ess_ratio': 1.0})
    with pytest.raises(ValueError, match="local_ess_ratio"):
        ui_main.smc_sampler(**{**base, 'local_ess_ratio': 0.0})


def test_parse_engine_rejects_bad_configs_early():
    with pytest.raises(ValueError, match="'name'"):
        parse_engine({})
    with pytest.raises(ValueError, match="Unknown engine"):
        parse_engine({'name': 'nuts'})
    with pytest.raises(ValueError, match="Invalid config"):
        parse_engine({'name': 'hmc', 'unknown': 1})


def test_parse_explorer_rejects_incomplete_nonadaptive_config():
    with pytest.raises(ValueError, match="'niter'"):
        parse_explorer('nonadaptive', mcmc_params=object())
    with pytest.raises(ValueError, match="'mcmc_params'"):
        parse_explorer('nonadaptive', niter=10)
    with pytest.raises(ValueError, match="Unknown explorer mode"):
        parse_explorer('bogus')


def test_parse_collectors_rejects_bad_configs_early():
    with pytest.raises(ValueError, match="'name'"):
        parse_collectors([{}])
    with pytest.raises(ValueError, match="Unknown collector"):
        parse_collectors([{'name': 'bogus'}])
    with pytest.raises(ValueError, match="Unknown logZ estimator"):
        parse_collectors([{'name': 'logZ', 'estimator': 'bogus'}])


def test_parse_swappers_is_keyword_only():
    with pytest.raises(TypeError):
        parse_swappers(True)
