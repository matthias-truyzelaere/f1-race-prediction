"""Tests for utils module."""

from utils import F1_POINTS_SYSTEM, calculate_points


class TestCalculatePoints:
    """Tests for calculate_points function."""

    def test_calculate_points_first_place_returns_25(self) -> None:
        """First place should return 25 points."""
        assert calculate_points(1) == 25

    def test_calculate_points_second_place_returns_18(self) -> None:
        """Second place should return 18 points."""
        assert calculate_points(2) == 18

    def test_calculate_points_third_place_returns_15(self) -> None:
        """Third place should return 15 points."""
        assert calculate_points(3) == 15

    def test_calculate_points_tenth_place_returns_1(self) -> None:
        """Tenth place should return 1 point."""
        assert calculate_points(10) == 1

    def test_calculate_points_outside_top_10_returns_0(self) -> None:
        """Positions outside top 10 should return 0 points."""
        assert calculate_points(11) == 0
        assert calculate_points(15) == 0
        assert calculate_points(20) == 0

    def test_calculate_points_nan_returns_0(self) -> None:
        """NaN position should return 0 points."""
        import numpy as np

        assert calculate_points(np.nan) == 0


class TestF1PointsSystem:
    """Tests for F1_POINTS_SYSTEM constant."""

    def test_points_system_has_10_positions(self) -> None:
        """Points system should have exactly 10 scoring positions."""
        assert len(F1_POINTS_SYSTEM) == 10

    def test_points_system_first_place_is_25(self) -> None:
        """First place in points system should be 25."""
        assert F1_POINTS_SYSTEM[1] == 25

    def test_points_system_total_is_correct(self) -> None:
        """Total points available should be 25+18+15+12+10+8+6+4+2+1=101."""
        assert sum(F1_POINTS_SYSTEM.values()) == 101
