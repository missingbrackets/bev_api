"""Tests for the CD rater adverse-weather migration.

Covers:
1. Batching boundaries (daily: 25 max, expanding: 10 max)
2. Single-thread wave pause behaviour (concurrency=1, 1 s pause)
3. 500 batch failure -> per-event isolation
4. Failed-events capture and return contract
5. Lat/lon inclusion in payload
6. Probability divide-by-100 normalisation
7. Output table shape compatibility with CD rater schema
8. Location mapping loading and filtering
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Ensure project root is importable (conftest.py already does this, but
# be explicit for IDE / direct invocation).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also add cd rater directory to path for relative imports within that package.
CD_RATER_ROOT = PROJECT_ROOT / "cd rater"
if str(CD_RATER_ROOT) not in sys.path:
    sys.path.insert(0, str(CD_RATER_ROOT))

from algorithms.tasks_section.adverse_weather import (
    DAILY_PERILS,
    EXPANDING_PERILS,
    _CD_BATCH_PAUSE_SECONDS,
    _CD_CONCURRENCY,
    build_events_daily,
    build_events_expanding,
    load_location_mapping,
    lookup_locations,
    normalise_daily_results,
    normalise_expanding_results,
    run_adverse_weather,
)
from bev_client import ENDPOINT_MAX_COMBINATIONS, batch_payload


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _sample_locations_df(n: int = 5) -> pd.DataFrame:
    """Return a small DataFrame of fake locations with lat/lon."""
    return pd.DataFrame({
        "country": [f"Country{i}" for i in range(n)],
        "area": [f"Area{i}" for i in range(n)],
        "city": [f"City{i}" for i in range(n)],
        "location_id": list(range(1000, 1000 + n)),
        "latitude": [51.5 + i * 0.1 for i in range(n)],
        "longitude": [-0.1 + i * 0.1 for i in range(n)],
    })


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeSession:
    """Session that records POSTs and returns prod-shaped daily responses."""
    def __init__(self):
        self.headers = {}
        self.verify = False
        self.calls: list[dict] = []

    def mount(self, prefix, adapter):
        pass

    def post(self, url, headers=None, data=None, timeout=None):
        payload = json.loads(data)
        self.calls.append(payload)
        events = payload.get("events", [])
        perils = payload.get("perils", [])
        # Return realistic prod-shaped response per event x peril.
        results = []
        for ev in events:
            for peril in perils:
                results.append({
                    "index": ev["index"],
                    "peril": peril,
                    "latitude": ev.get("latitude"),
                    "longitude": ev.get("longitude"),
                    "threshold": ["10", "50"],
                    "probability": [82.0, 45.0],
                    "unit": "mm",
                })
        return _FakeResponse(results)


class _FakeSessionExpanding:
    """Session that returns prod-shaped expanding responses."""
    def __init__(self):
        self.headers = {}
        self.verify = False
        self.calls: list[dict] = []

    def mount(self, prefix, adapter):
        pass

    def post(self, url, headers=None, data=None, timeout=None):
        payload = json.loads(data)
        self.calls.append(payload)
        events = payload.get("events", [])
        results = []
        for ev in events:
            results.append({
                "index": ev["index"],
                "window_index": 1,
                "peril": "CumulativeRain",
                "threshold": 50.0,
                "value": 75.0,
            })
        return _FakeResponse(results)


class _FakeSession500:
    """Session that fails on multi-event batches, succeeds on single-event
    (except index 2 which always fails)."""
    def __init__(self):
        self.headers = {}
        self.verify = False

    def mount(self, prefix, adapter):
        pass

    def post(self, url, headers=None, data=None, timeout=None):
        payload = json.loads(data)
        events = payload.get("events", [])
        perils = payload.get("perils", [])

        if len(events) > 1:
            raise Exception("500 Server Error: simulated batch failure")
        if events and events[0].get("index") == 2:
            raise Exception("500 Server Error: bad location index=2")

        ev = events[0]
        return _FakeResponse([{
            "index": ev["index"],
            "peril": perils[0] if perils else "Rain",
            "latitude": ev.get("latitude"),
            "longitude": ev.get("longitude"),
            "threshold": ["10"],
            "probability": [82.0],
            "unit": "mm",
        }])


class _TimingSession:
    """Session that records wall-clock time of each POST."""
    def __init__(self):
        self.headers = {}
        self.verify = False
        self.timestamps: list[float] = []

    def mount(self, prefix, adapter):
        pass

    def post(self, url, headers=None, data=None, timeout=None):
        self.timestamps.append(time.monotonic())
        payload = json.loads(data)
        events = payload.get("events", [])
        perils = payload.get("perils", [])
        results = []
        for ev in events:
            for peril in perils:
                results.append({
                    "index": ev["index"],
                    "peril": peril,
                    "threshold": ["10"],
                    "probability": [50.0],
                    "unit": "mm",
                })
        return _FakeResponse(results)


# ===================================================================
# 1. Batching boundaries
# ===================================================================

class TestBatchingBoundaries:
    """Verify event-peril combination limits per endpoint."""

    def test_daily_max_25_combinations(self):
        """With 4 perils and max_combinations=25, max 6 events per batch."""
        perils = DAILY_PERILS  # 4 perils
        events = [{"index": i} for i in range(20)]
        batches = batch_payload(
            perils, events,
            max_combinations=ENDPOINT_MAX_COMBINATIONS["daily"],
        )
        # 25 // 4 = 6 events per batch -> ceil(20/6) = 4 batches
        assert len(batches) == 4
        for b in batches:
            assert len(b["events"]) <= 6
        assert sum(len(b["events"]) for b in batches) == 20

    def test_expanding_max_10_combinations(self):
        """With 1 peril and max_combinations=10, max 10 events per batch."""
        perils = EXPANDING_PERILS  # 1 peril
        events = [{"index": i} for i in range(25)]
        batches = batch_payload(
            perils, events,
            window_days=0,
            max_combinations=ENDPOINT_MAX_COMBINATIONS["expanding"],
        )
        # 10 // 1 = 10 events per batch -> ceil(25/10) = 3 batches
        assert len(batches) == 3
        for b in batches:
            assert len(b["events"]) <= 10
            assert "window_days" in b
        assert sum(len(b["events"]) for b in batches) == 25

    def test_single_peril_daily(self):
        """With 1 peril, up to 25 events per batch."""
        perils = ["Rain"]
        events = [{"index": i} for i in range(30)]
        batches = batch_payload(perils, events, max_combinations=25)
        assert len(batches) == 2  # 25 + 5
        assert len(batches[0]["events"]) == 25
        assert len(batches[1]["events"]) == 5


# ===================================================================
# 2. Single-thread wave pause behaviour
# ===================================================================

class TestWavePauseBehaviour:
    """CD rater must use concurrency=1 with 1 s pause between waves."""

    def test_concurrency_constant_is_one(self):
        assert _CD_CONCURRENCY == 1

    def test_batch_pause_constant_is_one_second(self):
        assert _CD_BATCH_PAUSE_SECONDS == 1.0

    def test_wave_pause_between_payloads(self, monkeypatch):
        """With concurrency=1, each payload is its own wave.  Verify that
        successive POSTs are spaced ~1 s apart."""
        timing_session = _TimingSession()
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: timing_session,
        )

        locations_df = _sample_locations_df(3)
        # 1 peril, max_combinations=25 => all 3 events in 1 batch
        # Need >1 batch: use max_combinations=1 so each event is its own batch
        run_adverse_weather(
            locations_df=locations_df,
            perils_daily=["Rain"],
            perils_expanding=[],  # skip expanding
            api_key="test-key",
            base_url="https://example.com/api",
        )

        # With 3 events and 4 perils (default) at max_combinations=25,
        # 25//4=6 events per batch => 1 batch => no pause.
        # So we need to override to force multiple batches.
        # Let's redo with forced small max_combinations via monkeypatch.
        timing_session.timestamps.clear()

        # Force max_combinations to 1 so each event is its own batch.
        orig_endpoint_max = ENDPOINT_MAX_COMBINATIONS.copy()
        monkeypatch.setitem(ENDPOINT_MAX_COMBINATIONS, "daily", 1)

        run_adverse_weather(
            locations_df=locations_df,
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
        )

        # With 3 locations and max_combinations=1 (1 peril), each location
        # is its own batch => 3 payloads => 2 pauses.
        ts = timing_session.timestamps
        assert len(ts) == 3, f"Expected 3 POSTs, got {len(ts)}"

        # Verify gaps of ~1 s between consecutive posts
        for i in range(1, len(ts)):
            gap = ts[i] - ts[i - 1]
            assert gap >= 0.9, (
                f"Gap between POST {i-1} and {i} was {gap:.2f}s, expected >= 0.9s"
            )

        # Restore
        monkeypatch.setitem(ENDPOINT_MAX_COMBINATIONS, "daily", orig_endpoint_max["daily"])


# ===================================================================
# 3. 500 batch failure -> per-event isolation
# ===================================================================

class TestFailureIsolation:
    """When a batched request fails, events are retried one-by-one."""

    def test_batch_failure_isolates_bad_location(self, monkeypatch):
        """Multi-event batches fail; individual retries succeed except
        index 2.  Events 0, 1, 3 should succeed; event 2 should be
        captured in failed_events."""
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: _FakeSession500(),
        )

        locations_df = _sample_locations_df(4)

        daily_rows, expanding_rows, failed_events = run_adverse_weather(
            locations_df=locations_df,
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
            max_retries=0,
        )

        # 3 events succeed (0, 1, 3) with 1 threshold each -> 3 rows
        assert len(daily_rows) == 3
        successful_indices = {r["index"] for r in daily_rows}
        assert successful_indices == {0, 1, 3}

        # 1 event fails (index 2)
        assert len(failed_events) == 1
        assert failed_events[0]["event"]["index"] == 2
        assert "error" in failed_events[0]


# ===================================================================
# 4. Failed-events capture and return contract
# ===================================================================

class TestReturnContract:
    """Callers must be able to unpack (results, failed_events)."""

    def test_return_is_three_tuple(self, monkeypatch):
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: _FakeSession(),
        )
        result = run_adverse_weather(
            locations_df=_sample_locations_df(2),
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        daily_rows, expanding_rows, failed_events = result
        assert isinstance(daily_rows, list)
        assert isinstance(expanding_rows, list)
        assert isinstance(failed_events, list)

    def test_failed_event_structure(self, monkeypatch):
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: _FakeSession500(),
        )
        _, _, failed_events = run_adverse_weather(
            locations_df=_sample_locations_df(4),
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
            max_retries=0,
        )
        for fe in failed_events:
            assert "event" in fe, "Failed event must contain 'event' key"
            assert "error" in fe, "Failed event must contain 'error' key"
            assert isinstance(fe["event"], dict)
            assert isinstance(fe["error"], str)


# ===================================================================
# 5. Lat/lon inclusion in payload
# ===================================================================

class TestLatLonInPayload:
    """Events sent to the API must include latitude and longitude."""

    def test_daily_events_include_lat_lon(self):
        locations_df = _sample_locations_df(3)
        events, labels = build_events_daily(
            locations_df, "2025-01-01", "2025-01-01",
        )
        for ev in events:
            assert "latitude" in ev, "Daily event must include 'latitude'"
            assert "longitude" in ev, "Daily event must include 'longitude'"
            assert isinstance(ev["latitude"], float)
            assert isinstance(ev["longitude"], float)
            assert ev["location"] == "", "location field must be empty string"

    def test_expanding_events_include_lat_lon(self):
        locations_df = _sample_locations_df(3)
        events, labels = build_events_expanding(
            locations_df, "2025-01-01", "2025-01-01",
        )
        for ev in events:
            assert "latitude" in ev
            assert "longitude" in ev
            assert "tag" in ev
            assert "start_hour" in ev
            assert "end_hour" in ev
            assert ev["location"] == ""

    def test_payload_sent_to_api_has_lat_lon(self, monkeypatch):
        """Verify that the actual POST payload includes lat/lon."""
        fake_session = _FakeSession()
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: fake_session,
        )

        run_adverse_weather(
            locations_df=_sample_locations_df(2),
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
        )

        assert len(fake_session.calls) >= 1
        for call_payload in fake_session.calls:
            for ev in call_payload["events"]:
                assert "latitude" in ev
                assert "longitude" in ev


# ===================================================================
# 6. Probability divide-by-100 normalisation
# ===================================================================

class TestProbabilityNormalisation:
    """API returns probabilities on 0-100 scale; CD rater needs 0-1."""

    def test_daily_probability_divided_by_100(self):
        raw = [{
            "index": 0,
            "peril": "Rain",
            "threshold": ["10", "50"],
            "probability": [82.0, 45.0],
            "unit": "mm",
        }]
        rows = normalise_daily_results(raw)
        assert len(rows) == 2
        assert rows[0]["value"] == pytest.approx(0.82)
        assert rows[1]["value"] == pytest.approx(0.45)

    def test_daily_legacy_model_format_divided_by_100(self):
        raw = [{
            "index": 0,
            "peril": "Rain",
            "model": {"threshold_10": 82.0, "threshold_50": 45.0},
        }]
        rows = normalise_daily_results(raw)
        assert len(rows) == 2
        values = {r["threshold"]: r["value"] for r in rows}
        assert values["threshold_10"] == pytest.approx(0.82)
        assert values["threshold_50"] == pytest.approx(0.45)

    def test_expanding_value_divided_by_100(self):
        raw = [{
            "index": 0,
            "window_index": 1,
            "peril": "CumulativeRain",
            "threshold": 50.0,
            "value": 75.0,
        }]
        rows = normalise_expanding_results(raw)
        assert len(rows) == 1
        assert rows[0]["value"] == pytest.approx(0.75)
        assert rows[0]["threshold"] == pytest.approx(50.0)

    def test_normalisation_applied_exactly_once(self, monkeypatch):
        """End-to-end: probability 82 in API response becomes 0.82 in
        the returned daily_rows, not 0.0082 (double-scaled)."""
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: _FakeSession(),
        )

        daily_rows, _, _ = run_adverse_weather(
            locations_df=_sample_locations_df(1),
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
        )

        # _FakeSession returns probability=[82.0, 45.0]
        values = [r["value"] for r in daily_rows]
        assert 0.82 in [pytest.approx(v) for v in values]
        assert 0.45 in [pytest.approx(v) for v in values]
        # None should be < 0.01 (would indicate double-scaling)
        for v in values:
            assert v is None or v >= 0.01, f"Value {v} looks double-scaled"


# ===================================================================
# 7. Output table shape compatibility
# ===================================================================

class TestOutputShape:
    """Verify output matches the CD rater adverse_weather schema."""

    def test_daily_row_keys(self, monkeypatch):
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: _FakeSession(),
        )
        daily_rows, _, _ = run_adverse_weather(
            locations_df=_sample_locations_df(1),
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
        )
        expected_keys = {"peril", "index", "threshold", "value"}
        for row in daily_rows:
            assert set(row.keys()) == expected_keys, (
                f"Daily row keys mismatch: {set(row.keys())} != {expected_keys}"
            )

    def test_expanding_row_keys(self, monkeypatch):
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: _FakeSessionExpanding(),
        )
        _, expanding_rows, _ = run_adverse_weather(
            locations_df=_sample_locations_df(1),
            perils_daily=[],
            perils_expanding=["CumulativeRain"],
            api_key="test-key",
            base_url="https://example.com/api",
            window_days=0,
        )
        expected_keys = {"index", "window_index", "peril", "threshold", "value"}
        for row in expanding_rows:
            assert set(row.keys()) == expected_keys, (
                f"Expanding row keys mismatch: {set(row.keys())} != {expected_keys}"
            )

    def test_daily_rows_convert_to_dataframe(self, monkeypatch):
        """The output must be convertible to a DataFrame with the expected
        columns (as downstream consumers will do)."""
        monkeypatch.setattr(
            "bev_client.requests.Session", lambda: _FakeSession(),
        )
        daily_rows, _, _ = run_adverse_weather(
            locations_df=_sample_locations_df(2),
            perils_daily=["Rain"],
            perils_expanding=[],
            api_key="test-key",
            base_url="https://example.com/api",
        )
        df = pd.DataFrame(daily_rows)
        assert set(df.columns) == {"peril", "index", "threshold", "value"}
        assert df["value"].dtype in ("float64", "object")  # floats or mixed with None


# ===================================================================
# 8. Location mapping loading and filtering
# ===================================================================

class TestLocationMapping:
    """Test load_location_mapping and lookup_locations."""

    def test_load_mapping_has_lat_lon(self):
        """The new mapping CSV must contain latitude and longitude."""
        df = load_location_mapping()
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert "location_id" in df.columns
        assert len(df) > 0

    def test_lookup_by_country(self):
        df = load_location_mapping()
        filtered = lookup_locations(df, countries=["Afghanistan"])
        assert len(filtered) > 0
        assert all(filtered["country"] == "Afghanistan")

    def test_lookup_returns_empty_on_no_match(self):
        df = load_location_mapping()
        filtered = lookup_locations(df, countries=["NonexistentLand"])
        assert len(filtered) == 0

    def test_lookup_by_location_ids(self):
        df = load_location_mapping()
        sample_ids = df["location_id"].head(3).tolist()
        filtered = lookup_locations(df, location_ids=sample_ids)
        assert len(filtered) == 3

    def test_missing_csv_raises(self):
        with pytest.raises(FileNotFoundError):
            load_location_mapping("/nonexistent/path/to/file.csv")
