"""
Profile validation tests.

Tests that profile JSON files have correct structure and valid data.
Uses ONLY synthetic test profiles (no personal data).
"""

import pytest
import json
from pathlib import Path


@pytest.mark.profile
class TestProfileStructure:
    """Test profile JSON structure and required fields."""

    def test_profile_has_required_top_level_fields(
        self, sample_profile_default, profile_validation_config
    ):
        """Test that profile has all required top-level fields."""
        required_fields = profile_validation_config['required_fields']

        for field in required_fields:
            assert field in sample_profile_default, (
                f"Profile missing required field: {field}"
            )

    def test_profile_has_required_metric_fields(
        self, sample_profile_default, profile_validation_config
    ):
        """Test that profile.metrics has required nested fields."""
        required_metric_fields = profile_validation_config['required_metric_fields']

        assert 'metrics' in sample_profile_default
        metrics = sample_profile_default['metrics']

        for field in required_metric_fields:
            assert field in metrics, (
                f"Profile.metrics missing required field: {field}"
            )

    def test_profile_has_required_transition_categories(
        self, sample_profile_default, profile_validation_config
    ):
        """Test that profile has required transition categories."""
        required_categories = profile_validation_config['required_transition_categories']

        assert 'transitions' in sample_profile_default
        transitions = sample_profile_default['transitions']

        for category in required_categories:
            assert category in transitions, (
                f"Profile missing required transition category: {category}"
            )

    def test_excellence_profile_has_additional_categories(
        self, sample_profile_excellence, profile_validation_config
    ):
        """Test that excellence profile has additional transition categories."""
        additional_categories = profile_validation_config['excellence_additional_categories']

        assert 'transitions' in sample_profile_excellence
        transitions = sample_profile_excellence['transitions']

        for category in additional_categories:
            assert category in transitions, (
                f"Excellence profile missing additional category: {category}"
            )


@pytest.mark.profile
class TestProfileDataValidity:
    """Test that profile data values are valid."""

    def test_analyzed_files_is_positive(self, sample_profile_default):
        """Test that analyzed_files count is positive."""
        assert sample_profile_default['analyzed_files'] > 0

    def test_primary_language_is_valid(self, sample_profile_default):
        """Test that primary_language is a valid code."""
        valid_languages = ['de', 'en']
        assert sample_profile_default['primary_language'] in valid_languages

    def test_metrics_basic_values_are_positive(self, sample_profile_default):
        """Test that basic metrics have positive values."""
        basic = sample_profile_default['metrics']['basic']

        assert basic['total_words'] > 0, "total_words must be positive"
        assert basic['total_sentences'] > 0, "total_sentences must be positive"
        assert basic['avg_sentence_length'] > 0, "avg_sentence_length must be positive"

    def test_lexical_diversity_in_valid_range(self, sample_profile_default):
        """Test that lexical diversity is between 0 and 1."""
        lexical_div = sample_profile_default['metrics']['basic']['lexical_diversity']
        assert 0 < lexical_div <= 1, (
            f"lexical_diversity must be in (0, 1], got {lexical_div}"
        )

    def test_passive_ratio_in_valid_range(self, sample_profile_default):
        """Test that passive_ratio is between 0 and 1."""
        passive_ratio = sample_profile_default['metrics']['voice_and_style']['passive_ratio']
        assert 0 <= passive_ratio <= 1, (
            f"passive_ratio must be in [0, 1], got {passive_ratio}"
        )

    def test_transition_categories_have_patterns(self, sample_profile_default):
        """Test that each transition category has at least one pattern."""
        transitions = sample_profile_default['transitions']

        for category, patterns in transitions.items():
            assert isinstance(patterns, list), (
                f"Transition category '{category}' must be a list"
            )
            assert len(patterns) > 0, (
                f"Transition category '{category}' must have at least one pattern"
            )

    def test_transition_patterns_are_strings(self, sample_profile_default):
        """Test that all transition patterns are non-empty strings."""
        transitions = sample_profile_default['transitions']

        for category, patterns in transitions.items():
            for pattern in patterns:
                assert isinstance(pattern, str), (
                    f"Pattern in '{category}' must be string, got {type(pattern)}"
                )
                assert len(pattern.strip()) > 0, (
                    f"Pattern in '{category}' cannot be empty"
                )


@pytest.mark.profile
class TestProfileConsistency:
    """Test internal consistency of profile data."""

    def test_avg_sentence_length_matches_totals(self, sample_profile_default):
        """Test that avg_sentence_length is consistent with totals."""
        basic = sample_profile_default['metrics']['basic']

        expected_avg = basic['total_words'] / basic['total_sentences']
        actual_avg = basic['avg_sentence_length']

        # Allow small floating-point tolerance
        diff = abs(expected_avg - actual_avg)
        assert diff < 0.5, (
            f"avg_sentence_length inconsistent: "
            f"expected {expected_avg:.2f}, got {actual_avg}"
        )

    def test_transition_density_matches_totals(self, sample_profile_default):
        """Test that transition_density is consistent with total_transitions."""
        metrics = sample_profile_default['metrics']
        total_transitions = metrics['transitions']['total_transitions']
        transition_density = metrics['transitions']['transition_density']
        total_words = metrics['basic']['total_words']

        expected_density = (total_transitions / total_words) * 100
        diff = abs(expected_density - transition_density)

        assert diff < 0.5, (
            f"transition_density inconsistent: "
            f"expected {expected_density:.2f}, got {transition_density}"
        )

    def test_category_counts_sum_to_total(self, sample_profile_default):
        """Test that category transition counts sum to total."""
        transitions_metrics = sample_profile_default['metrics']['transitions']
        by_category = transitions_metrics['by_category']

        category_sum = sum(cat['count'] for cat in by_category.values())
        total = transitions_metrics['total_transitions']

        assert category_sum == total, (
            f"Category counts ({category_sum}) don't sum to total ({total})"
        )


@pytest.mark.profile
class TestProfileComparison:
    """Test differences between default and excellence profiles."""

    def test_excellence_has_more_categories(
        self, sample_profile_default, sample_profile_excellence
    ):
        """Test that excellence profile has more transition categories."""
        default_cats = len(sample_profile_default['transitions'])
        excellence_cats = len(sample_profile_excellence['transitions'])

        assert excellence_cats > default_cats, (
            f"Excellence should have more categories: "
            f"default={default_cats}, excellence={excellence_cats}"
        )

    def test_excellence_has_higher_transition_density(
        self, sample_profile_default, sample_profile_excellence
    ):
        """Test that excellence profile has higher transition density."""
        default_density = sample_profile_default['metrics']['transitions']['transition_density']
        excellence_density = sample_profile_excellence['metrics']['transitions']['transition_density']

        assert excellence_density >= default_density, (
            f"Excellence should have >= transition density: "
            f"default={default_density}, excellence={excellence_density}"
        )

    def test_excellence_has_lower_passive_ratio(
        self, sample_profile_default, sample_profile_excellence
    ):
        """Test that excellence profile has lower passive voice ratio."""
        default_passive = sample_profile_default['metrics']['voice_and_style']['passive_ratio']
        excellence_passive = sample_profile_excellence['metrics']['voice_and_style']['passive_ratio']

        assert excellence_passive < default_passive, (
            f"Excellence should have lower passive ratio: "
            f"default={default_passive}, excellence={excellence_passive}"
        )
