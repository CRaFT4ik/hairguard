"""Tests for how the trained model and the existing state machine compose.

The model emits one probability per 20-frame window. The state machine
already requires sustained detection before it alerts and holds the alert
through a dismissal delay. The design under test here is that the model
*replaces the per-frame geometric verdict* feeding that machine, and changes
nothing else: the hysteresis stays, the zone gate stays, the episode
accounting stays.

What the tests pin down, in order of how much it would cost to get wrong:

  - the rule keeps working when there is no model;
  - a model that raises becomes a skipped window, not a dead detector;
  - a model that keeps raising hands back to the rule rather than going
    quiet forever;
  - the window is fed the hand the analyzer actually picked;
  - nothing is scored below the configured minimum;
  - the alert arrives sooner than waiting for a full window would allow,
    which is the whole reason `min_frames` exists.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pytest

from hairguard.analyzer import GestureAnalyzer
from hairguard.live_features import frame_record
from hairlab.features import FEATURE_DIM

from tests.test_analyzer_v2 import (
    _make_face,
    _make_open_hand_in_zone,
    _make_pinching_hand,
    _make_pose,
)

SEQ = 4  # short windows keep the tests readable; the slide is covered elsewhere


class _Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float = 0.15) -> None:
        self.now += seconds


class _FakeConfig:
    """Config stub covering the keys the analyzer reads, model keys included."""

    def __init__(self, **model: object) -> None:
        self._values = {
            ("detection", "trigger_delay_sec"): 0.8,
            ("detection", "dismiss_delay_sec"): 2.5,
            ("detection", "min_confidence"): 0.5,
            ("detection", "combined_threshold"): 0.55,
            ("detection", "rotation_threshold"): 0.10,
            ("detection", "persist_frames"): 1,
            ("detection", "min_presence_score"): 0.7,
            ("detection", "hair_zone_offset"): 0.4,
            ("detection", "model", "enabled"): True,
            ("detection", "model", "threshold"): "auto",
            ("detection", "model", "trigger_delay_sec"): 0.3,
            # The full window, so a test that cares about partial scoring has
            # to ask for it rather than inherit it. The shipped default is 3;
            # see tests/test_config.py and hairlab/truncation.py.
            ("detection", "model", "min_frames"): SEQ,
        }
        for key, value in model.items():
            self._values[("detection", "model", key)] = value

    def get(self, *keys: str):
        return self._values[keys]


class _FakeModel:
    """Stands in for the ONNX runtime: one probability per window."""

    def __init__(
        self,
        score: float | Callable[[np.ndarray], float] = 0.99,
        *,
        threshold: float = 0.5,
        seq_len: int = SEQ,
        raises: Optional[BaseException] = None,
    ) -> None:
        self.threshold = threshold
        self.seq_len = seq_len
        self.feature_dim = FEATURE_DIM
        self._score = score
        self._raises = raises
        self.windows: list[np.ndarray] = []

    def score(self, window: np.ndarray) -> float:
        self.windows.append(np.asarray(window).copy())
        if self._raises is not None:
            raise self._raises
        if callable(self._score):
            return float(self._score(window))
        return float(self._score)


def _analyzer(model: Optional[_FakeModel], clock: _Clock, **config) -> GestureAnalyzer:
    return GestureAnalyzer(_FakeConfig(**config), twist_model=model, clock=clock)


def _run(
    analyzer: GestureAnalyzer,
    clock: _Clock,
    hands,
    frames: int,
    step: float = 0.15,
):
    """Feed ``frames`` ticks of the same hands, returning the last result."""
    result = None
    pose, face = _make_pose(), _make_face()
    for _ in range(frames):
        result = analyzer.analyze(pose, list(hands), face, 480, 640)
        clock.tick(step)
    return result


# ---------------------------------------------------------------------------
# Without a model, nothing changes
# ---------------------------------------------------------------------------

class TestRuleStillWorks:
    def test_no_model_means_the_rule_decides(self) -> None:
        clock = _Clock()
        analyzer = _analyzer(None, clock)
        result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], 1)
        assert result.detector == "rule"
        assert result.hand_near_head is True
        assert result.model_score is None

    def test_the_rule_reaches_an_alert_on_its_own(self) -> None:
        clock = _Clock()
        analyzer = _analyzer(None, clock)
        result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], 12)
        assert result.state == "ALERTING"
        assert result.should_alert is True


# ---------------------------------------------------------------------------
# With a model, the model decides — but only once it can
# ---------------------------------------------------------------------------

class TestModelDecides:
    def test_nothing_fires_below_the_configured_minimum(self) -> None:
        """Here the minimum is the whole window; the shipped default is 3."""
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock)
        for index in range(SEQ - 1):
            result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], 1)
            assert result.hand_near_head is False, f"fired at frame {index}"
            assert result.model_score is None
        assert model.windows == []

    def test_a_confident_window_starts_the_state_machine(self) -> None:
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock)
        result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ)
        assert result.detector == "model"
        assert result.model_score == pytest.approx(0.99)
        assert result.hand_near_head is True
        assert result.state == "TRIGGERED"

    def test_a_sustained_window_reaches_the_alert(self) -> None:
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(score=0.99), clock)
        hands = [_make_pinching_hand(in_zone=True)]
        pose, face = _make_pose(), _make_face()

        results = []
        for _ in range(SEQ + 4):
            results.append(analyzer.analyze(pose, list(hands), face, 480, 640))
            clock.tick(0.15)

        assert results[-1].state == "ALERTING"
        assert results[-1].should_alert is True
        # Exactly one episode opens, on the tick that enters ALERTING — the
        # statistics depend on this not firing repeatedly.
        assert sum(r.episode_started for r in results) == 1

    def test_the_model_can_veto_what_the_rule_would_have_alerted_on(self) -> None:
        """Precision: the geometry says twisting, the model says resting."""
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(score=0.01), clock)
        result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], 12)
        assert result.model_score == pytest.approx(0.01)
        assert result.hand_near_head is False
        assert result.state == "IDLE"

    def test_the_model_can_alert_on_what_the_rule_would_have_missed(self) -> None:
        """Recall: this is the whole reason for the model.

        An open hand in the zone never satisfies the pinch-and-spread rule,
        so the old detector could not fire here at all.
        """
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(score=0.99), clock)
        result = _run(analyzer, clock, [_make_open_hand_in_zone()], SEQ + 4)
        assert result.state == "ALERTING"
        assert result.should_alert is True

    def test_a_score_exactly_on_the_threshold_counts_as_twisting(self) -> None:
        """``>=``, matching how the threshold was chosen during training."""
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(score=0.5, threshold=0.5), clock)
        result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ)
        assert result.hand_near_head is True

    def test_the_configured_trigger_delay_applies_to_the_model_path(self) -> None:
        """The window is already two seconds of evidence, so this one is short.

        Stacking the rule's 0.8 s on top of a full window would put the
        alert nearly four seconds after the hand arrived.
        """
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(score=0.99), clock, trigger_delay_sec=0.3)
        hands = [_make_pinching_hand(in_zone=True)]

        _run(analyzer, clock, hands, SEQ, step=0.05)   # window fills, TRIGGERED
        early = _run(analyzer, clock, hands, 1, step=0.05)
        assert early.state == "TRIGGERED"

        clock.tick(0.4)
        late = _run(analyzer, clock, hands, 1)
        assert late.state == "ALERTING"


# ---------------------------------------------------------------------------
# The window is built from what the loop actually saw
# ---------------------------------------------------------------------------

class TestWindowContents:
    def test_the_window_holds_the_hand_the_analyzer_picked(self) -> None:
        """Two hands visible; the clip and the window must agree on which.

        Picking the other one would train on one hand and infer on another,
        and the only symptom would be a worse detector.
        """
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock)

        out_of_zone = _make_pinching_hand(in_zone=False)
        in_zone = _make_pinching_hand(in_zone=True)
        pose, face = _make_pose(), _make_face()

        times = []
        for _ in range(SEQ):
            times.append(clock.now)
            result = analyzer.analyze(pose, [out_of_zone, in_zone], face, 480, 640)
            clock.tick(0.15)

        assert result.in_zone_hand_index == 1

        from hairlab.features import clip_features

        records = [frame_record(in_zone, pose, t=t) for t in times]
        assert np.array_equal(model.windows[0], clip_features(records, seq_len=SEQ))

    def test_leaving_the_zone_discards_the_window(self) -> None:
        """One unbroken visit per window, exactly as the clips were recorded."""
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock)
        hands = [_make_pinching_hand(in_zone=True)]

        _run(analyzer, clock, hands, SEQ - 1)
        _run(analyzer, clock, [], 1)                       # hand gone
        result = _run(analyzer, clock, hands, SEQ - 1)     # not enough on its own

        assert model.windows == []
        assert result.hand_near_head is False

    def test_a_dropped_hand_frame_restarts_the_window(self) -> None:
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock)
        hands = [_make_pinching_hand(in_zone=True)]

        _run(analyzer, clock, hands, SEQ)
        assert len(model.windows) == 1
        _run(analyzer, clock, [], 1)
        _run(analyzer, clock, hands, SEQ - 1)
        assert len(model.windows) == 1, "scored a window stitched across a gap"

    def test_the_window_keeps_sliding_while_the_hand_stays(self) -> None:
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock)
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ + 3)
        assert len(model.windows) == 4


# ---------------------------------------------------------------------------
# Failure degrades, never crashes
# ---------------------------------------------------------------------------

class TestFailureDegrades:
    def test_a_raising_model_does_not_take_the_tick_down(self, capsys) -> None:
        clock = _Clock()
        model = _FakeModel(raises=ValueError("non-finite feature"))
        analyzer = _analyzer(model, clock)
        result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ)
        assert result.hand_near_head is False
        assert result.model_score is None
        assert "non-finite feature" in capsys.readouterr().out

    def test_the_failure_is_logged_once_not_every_tick(self, capsys) -> None:
        """At 10 fps a per-tick log fills the console and hides everything else."""
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(raises=ValueError("boom")), clock)
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ)
        capsys.readouterr()
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], 3)
        assert capsys.readouterr().out == ""

    def test_a_single_failure_does_not_kill_the_detector(self) -> None:
        """One bad window must not cost the rest of the session."""
        clock = _Clock()
        scores = iter([None] + [0.99] * 20)

        class Flaky(_FakeModel):
            def score(self, window):
                value = next(scores)
                if value is None:
                    raise ValueError("one bad window")
                return value

        analyzer = _analyzer(Flaky(), clock)
        result = _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ + 6)
        assert result.detector == "model"
        assert result.state == "ALERTING"

    def test_a_model_that_always_raises_hands_back_to_the_rule(self, capsys) -> None:
        """A detector that scores nothing is worse than the rule it replaced."""
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(raises=ValueError("broken")), clock)
        hands = [_make_pinching_hand(in_zone=True)]

        _run(analyzer, clock, hands, SEQ + 40)
        assert analyzer.detector_name == "rule"
        assert "правил" in capsys.readouterr().out

        # And the rule actually works from there on.
        result = _run(analyzer, clock, hands, 12)
        assert result.detector == "rule"
        assert result.state == "ALERTING"


# ---------------------------------------------------------------------------
# Latency: the model is a passenger on the loop that owns the alert
# ---------------------------------------------------------------------------

class TestLatency:
    def test_a_slow_model_is_put_aside_rather_than_stuttering_the_alert(
        self, capsys
    ) -> None:
        """The detection tick also drives tkinter, the overlay and the sound."""
        import time as _time

        clock = _Clock()

        class Slow(_FakeModel):
            def score(self, window):
                _time.sleep(0.04)
                return 0.99

        analyzer = _analyzer(Slow(), clock)
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ + 12)

        assert analyzer.detector_name == "rule"
        output = capsys.readouterr().out
        assert "мс" in output
        assert "правил" in output

    def test_an_occasional_slow_window_is_tolerated(self) -> None:
        """A scheduler hiccup is not a reason to give up thirty points of recall."""
        import time as _time

        clock = _Clock()
        calls = {"n": 0}

        class Hiccup(_FakeModel):
            def score(self, window):
                calls["n"] += 1
                if calls["n"] % 4 == 0:
                    _time.sleep(0.04)
                return 0.99

        analyzer = _analyzer(Hiccup(), clock)
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ + 20)
        assert analyzer.detector_name == "model"

    def test_the_worst_window_is_reported(self) -> None:
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(score=0.99), clock)
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], SEQ)
        assert analyzer.model_worst_sec > 0.0


@pytest.mark.skipif(
    not (__import__("pathlib").Path.home() / ".hairguard" / "trained"
         / "twistnet.onnx").is_file(),
    reason="no exported model to measure",
)
class TestRealModelCost:
    """What the real model actually costs inside the detection tick.

    A measurement, kept as a test so it cannot quietly regress. The budget
    is the tripwire the analyzer enforces; the assertion below is much
    tighter, because the point is to know there is real headroom rather than
    to discover the limit was reached.
    """

    def test_scoring_a_window_costs_a_fraction_of_a_tick(self) -> None:
        import time as _time
        from pathlib import Path

        from hairguard.twistnet import TwistModel

        model = TwistModel.load(Path.home() / ".hairguard" / "trained")
        clock = _Clock()
        analyzer = _analyzer(None, clock)
        analyzer._twist = model
        from hairguard.live_features import FeatureWindow

        analyzer._feature_window = FeatureWindow(seq_len=model.seq_len)

        hands = [_make_pinching_hand(in_zone=True)]
        try:
            _run(analyzer, clock, hands, model.seq_len)      # fill, then warm
            _run(analyzer, clock, hands, 20)

            started = _time.perf_counter()
            _run(analyzer, clock, hands, 100)
            per_tick = (_time.perf_counter() - started) / 100
        finally:
            model.close()

        # The tick budget is 100 ms at the configured 10 fps, and three
        # MediaPipe models already spend most of it.
        assert per_tick < 0.010, f"{1000 * per_tick:.2f} ms per tick"


# ---------------------------------------------------------------------------
# Reaction time: how soon the model gets asked
# ---------------------------------------------------------------------------

class TestReactionTime:
    """The app interrupts a behaviour, so waiting is the thing it cannot do.

    ``hairlab.truncation`` measured the model on repeat-padded partial
    windows, cross-validated by capture day: recall 78.0% +/- 8.2 at three
    frames against 86.7% +/- 6.8 at twenty, with the false-alarm rate on
    "hand resting on head" slightly better at three (3.2% against 3.6%). The
    rule this replaces manages 53.9%. So asking early is nearly free, and
    waiting nearly three seconds for a full window is not.
    """

    def test_the_model_is_asked_as_soon_as_the_minimum_is_reached(self) -> None:
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock, min_frames=2)
        hands = [_make_pinching_hand(in_zone=True)]

        _run(analyzer, clock, hands, 1)
        assert model.windows == [], "scored before the minimum"
        result = _run(analyzer, clock, hands, 1)
        assert len(model.windows) == 1
        assert result.model_score == pytest.approx(0.99)

    def test_a_partial_window_is_still_the_full_tensor_shape(self) -> None:
        """The graph takes a fixed window; padding is what makes that work."""
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock, min_frames=2)
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], 2)
        assert model.windows[0].shape == (SEQ, FEATURE_DIM)

    def test_no_padded_window_carries_a_zero_row(self) -> None:
        """Zero rows blind this model completely — AUC 0.500. See truncation.py."""
        clock = _Clock()
        model = _FakeModel(score=0.99)
        analyzer = _analyzer(model, clock, min_frames=1)
        _run(analyzer, clock, [_make_pinching_hand(in_zone=True)], 3)
        for window in model.windows:
            assert int((~window.any(axis=1)).sum()) == 0

    def test_the_alert_arrives_sooner_than_waiting_for_a_full_window(self) -> None:
        """The whole point of the measurement, as an assertion."""
        hands = [_make_pinching_hand(in_zone=True)]

        def ticks_to_alert(min_frames: int) -> int:
            clock = _Clock()
            analyzer = _analyzer(_FakeModel(score=0.99), clock, min_frames=min_frames)
            pose, face = _make_pose(), _make_face()
            for tick in range(1, 40):
                result = analyzer.analyze(pose, list(hands), face, 480, 640)
                clock.tick(0.15)
                if result.should_alert:
                    return tick
            raise AssertionError(f"no alert with min_frames={min_frames}")

        early = ticks_to_alert(2)
        full = ticks_to_alert(SEQ)
        assert early < full, f"early={early} full={full}"

    def test_a_short_visit_can_still_be_caught(self) -> None:
        """4% of recorded twisting never reached twenty in-zone frames even
        inside a three-second clip — the short episodes, where reacting late
        means not reacting at all.

        Set up at the real window length and the shipped settings, because
        the question is whether those numbers leave room, not whether the
        mechanism works in the abstract.
        """
        clock = _Clock()
        model = _FakeModel(score=0.99, seq_len=20)
        analyzer = _analyzer(model, clock, min_frames=3, trigger_delay_sec=0.3)
        hands = [_make_pinching_hand(in_zone=True)]
        pose, face = _make_pose(), _make_face()

        alerted_at = None
        for tick in range(1, 20):         # never reaches a full window
            result = analyzer.analyze(pose, list(hands), face, 480, 640)
            if result.should_alert and alerted_at is None:
                alerted_at = tick
            clock.tick(1 / 6.75)          # the fps this loop actually achieves

        assert alerted_at is not None, "a visit shorter than the window never alerted"
        # Three frames to fill the minimum plus 0.3 s of agreement — about
        # 0.74 s in, against 2.81 s to fill the window at all.
        assert alerted_at <= 6, f"alerted at tick {alerted_at}"

    def test_the_minimum_is_clamped_to_the_window(self) -> None:
        clock = _Clock()
        analyzer = _analyzer(_FakeModel(score=0.99), clock, min_frames=500)
        assert analyzer._feature_window.min_frames == SEQ

    def test_an_unreadable_minimum_falls_back_to_the_full_window(self) -> None:
        """The conservative direction: slower, never less measured."""
        clock = _Clock()

        class Partial(_FakeConfig):
            def get(self, *keys):
                if keys == ("detection", "model", "min_frames"):
                    raise KeyError(keys)
                return super().get(*keys)

        analyzer = GestureAnalyzer(Partial(), twist_model=_FakeModel(), clock=clock)
        assert analyzer._feature_window.min_frames == SEQ
