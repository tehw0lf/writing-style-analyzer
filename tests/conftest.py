"""
Pytest configuration and shared fixtures for test suite.

This module provides common fixtures and configuration for all tests.
Uses ONLY synthetic test data from tests/fixtures/ (no personal data).
"""

import pytest
import yaml
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from german_academic_analyzer import analyze_text, load_profile, compare_to_profile


# ============================================================================
# Path Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to synthetic test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def baselines_dir():
    """Path to baselines directory."""
    return Path(__file__).parent / "baselines"


# ============================================================================
# Profile Fixtures (Synthetic Profiles)
# ============================================================================

@pytest.fixture(scope="session")
def sample_profile_default(fixtures_dir):
    """Load synthetic default academic profile."""
    profile_path = fixtures_dir / "sample_profile_default.json"
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_profile_excellence(fixtures_dir):
    """Load synthetic excellence academic profile."""
    profile_path = fixtures_dir / "sample_profile_excellence.json"
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Text Fixtures (Synthetic German Academic Texts)
# ============================================================================

@pytest.fixture(scope="session")
def sample_text_short(fixtures_dir):
    """Load synthetic short German academic text (~150 words)."""
    text_path = fixtures_dir / "sample_text_short.txt"
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture(scope="session")
def sample_text_long(fixtures_dir):
    """Load synthetic long German academic text (~300 words)."""
    text_path = fixtures_dir / "sample_text_long.txt"
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================================
# Tolerance Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def tolerances(test_config):
    """Get metric tolerances from config."""
    return test_config['tolerances']


@pytest.fixture(scope="session")
def profile_validation_config(test_config):
    """Get profile validation requirements from config."""
    return test_config['profile_validation']


# ============================================================================
# Helper Functions (Available to All Tests)
# ============================================================================

def assert_within_tolerance(actual, expected, tolerance, metric_name):
    """
    Assert that a metric is within acceptable tolerance.

    Args:
        actual: Actual value
        expected: Expected value
        tolerance: Allowed deviation
        metric_name: Name of metric (for error messages)
    """
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"{metric_name} outside tolerance: "
        f"expected {expected} ± {tolerance}, got {actual} (diff: {diff:.2f})"
    )


def assert_percentage_within_tolerance(actual, expected, tolerance_pct, metric_name):
    """
    Assert that a metric is within acceptable percentage tolerance.

    Args:
        actual: Actual value
        expected: Expected value
        tolerance_pct: Allowed percentage deviation
        metric_name: Name of metric (for error messages)
    """
    if expected == 0:
        assert actual == 0, f"{metric_name}: expected 0, got {actual}"
        return

    diff_pct = abs((actual - expected) / expected * 100)
    assert diff_pct <= tolerance_pct, (
        f"{metric_name} outside tolerance: "
        f"expected {expected} ± {tolerance_pct}%, got {actual} ({diff_pct:.1f}% diff)"
    )


# Make helper functions available as fixtures
@pytest.fixture
def assert_within_tol():
    """Fixture providing assert_within_tolerance function."""
    return assert_within_tolerance


@pytest.fixture
def assert_percentage_within_tol():
    """Fixture providing assert_percentage_within_tolerance function."""
    return assert_percentage_within_tolerance
