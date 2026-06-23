import pytest
from mppt.jax_configs import _set_allow_runtime_check

def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False,
                     help="include tests marked @pytest.mark.slow")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip unless --slow)")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test, use --slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

@pytest.fixture(autouse=True)
def reset_runtime_check():
    """Reset ALLOW_RUNTIME_CHECK to False before and after each test.
    test_logZ.py sets it to True at import time, which poisons the
    process for all subsequent tests.
    """
    _set_allow_runtime_check(False)
    yield
    _set_allow_runtime_check(False)
