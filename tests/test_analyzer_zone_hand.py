"""The analyzer reports which hand was found in the hair zone.

Dataset clips are cropped around that hand, so "some hand is in the zone" is
not enough — the collector needs to know which one. Fixtures are reused from
the main analyzer test module.
"""
from __future__ import annotations

from hairguard.analyzer import GestureAnalyzer

from tests.test_analyzer_v2 import _FakeConfig, _make_face, _make_hand, _make_pose


def _analyze(hands):
    analyzer = GestureAnalyzer(_FakeConfig())
    return analyzer.analyze(_make_pose(), hands, _make_face(), 480, 640)


def _hand_in_zone():
    """Fingertips above the forehead line at y=0.35."""
    return _make_hand(thumb_y=0.20, index_y=0.20)


def _hand_away():
    """Fingertips well below the forehead — nowhere near the hair."""
    return _make_hand(
        thumb_x=0.5, thumb_y=0.90, index_x=0.5, index_y=0.90,
        wrist_x=0.5, wrist_y=0.99, mcp_x=0.5, mcp_y=0.95,
    )


def test_no_hands_reports_no_index() -> None:
    assert _analyze([]).in_zone_hand_index is None


def test_hand_away_from_the_head_reports_no_index() -> None:
    result = _analyze([_hand_away()])
    assert result.fingertips_in_zone is False
    assert result.in_zone_hand_index is None


def test_the_only_hand_in_zone_is_reported() -> None:
    result = _analyze([_hand_in_zone()])
    assert result.fingertips_in_zone is True
    assert result.in_zone_hand_index == 0


def test_the_second_hand_is_reported_when_the_first_is_away() -> None:
    """With two hands tracked, the crop must follow the one at the head."""
    result = _analyze([_hand_away(), _hand_in_zone()])
    assert result.fingertips_in_zone is True
    assert result.in_zone_hand_index == 1
