__version__ = "0.1.1"

import pytest

from econsa.config import ROOT_DIR


def test(*args, **kwargs):
    """Run basic tests of the package."""
    pytest.main([str(ROOT_DIR), *args], **kwargs)
