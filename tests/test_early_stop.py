from __future__ import annotations


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


def test_time_based_early_stopping_triggers():
    from snake7.callbacks import WallTimeEarlyStopCallback

    clock = FakeClock(t=0.0)
    cb = WallTimeEarlyStopCallback(
        patience_seconds=1.0,
        window_size=1,
        min_delta=0.0,
        time_fn=clock.now,
    )

    cb.locals = {"infos": [{"episode": {"r": 1.0}}]}
    assert cb._on_step() is True

    clock.advance(2.0)
    cb.locals = {"infos": [{"episode": {"r": 1.0}}]}
    assert cb._on_step() is False


def test_patience_resets_on_improvement():
    from snake7.callbacks import WallTimeEarlyStopCallback

    clock = FakeClock(t=0.0)
    cb = WallTimeEarlyStopCallback(
        patience_seconds=1.0,
        window_size=1,
        min_delta=0.0,
        time_fn=clock.now,
    )

    cb.locals = {"infos": [{"episode": {"r": 1.0}}]}
    assert cb._on_step() is True

    clock.advance(0.9)
    cb.locals = {"infos": [{"episode": {"r": 2.0}}]}
    assert cb._on_step() is True

    clock.advance(0.9)
    cb.locals = {"infos": [{"episode": {"r": 2.0}}]}
    assert cb._on_step() is True

    clock.advance(0.2)
    cb.locals = {"infos": [{"episode": {"r": 2.0}}]}
    assert cb._on_step() is False


class _FakeModel:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        self.saved.append(str(path))


def test_best_model_is_saved_on_improvement(tmp_path):
    from snake7.callbacks import WallTimeEarlyStopCallback

    best_path = tmp_path / "best_model.zip"
    cb = WallTimeEarlyStopCallback(
        patience_seconds=999.0,
        window_size=1,
        min_delta=0.0,
        time_fn=lambda: 0.0,
        save_best_path=best_path,
    )
    cb.model = _FakeModel()

    cb.locals = {"infos": [{"episode": {"r": 1.0}}]}
    assert cb._on_step() is True
    assert cb.model.saved == [str(best_path)]

    cb.locals = {"infos": [{"episode": {"r": 1.0}}]}
    assert cb._on_step() is True
    assert cb.model.saved == [str(best_path)]


def test_stop_score_threshold_stops_training():
    from snake7.callbacks import WallTimeEarlyStopCallback

    cb = WallTimeEarlyStopCallback(
        patience_seconds=999.0,
        window_size=1,
        min_delta=0.0,
        time_fn=lambda: 0.0,
        stop_score_threshold=0.0,
    )
    cb.locals = {"infos": [{"episode": {"r": 1.0}}]}
    assert cb._on_step() is False
    assert cb.stopped is True
