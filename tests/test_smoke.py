"""
Smoke tests for Sensor Analytics Dashboard
"""
import pytest


def test_requirements_importable():
    """Core dependencies can be imported."""
    import pandas as pd
    import numpy as np
    import plotly
    import sqlalchemy
    assert pd.__version__
    assert np.__version__


def test_config_exists():
    """Config file exists and is valid YAML."""
    import yaml, os
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None


def test_src_structure_exists():
    """Core source directories exist."""
    import os
    assert os.path.exists("src")
    assert os.path.exists("app.py")