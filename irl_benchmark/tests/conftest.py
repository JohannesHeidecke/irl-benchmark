import pytest
import inspect

FAST_RUN = True


def pytest_addoption(parser):
    ''' Add run option '''
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests")


def pytest_runtest_setup(item):
    print("*** value of --runslow option: %r" %
          item.config.getoption("--runslow"))


def pytest_collection_modifyitems(config, items):
    ''' Skip tests marked as slow '''
    if not config.getoption("--runslow") and FAST_RUN:
        # Skip marked tests:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
