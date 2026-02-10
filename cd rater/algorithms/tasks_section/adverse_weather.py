"""Adverse-weather API workflow for the CD rater.

Mirrors the BEV client batching / error-isolation methodology
(``bev_client.bev_task_batches_threaded``) with CD-rater-specific
defaults:

* **concurrency = 1** (single-threaded payload execution)
* **1 s pause after each wave** (effectively after every payload)
* Automatic per-event retry on batch failure (500 isolation)
* Probability values normalised by dividing by 100

The public entry point is :func:`run_adverse_weather`.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bev_client import bev_task_batches_threaded, ENDPOINT_MAX_COMBINATIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CD rater forces single-threaded execution with a 1 s inter-wave pause.
_CD_CONCURRENCY = 1
_CD_BATCH_PAUSE_SECONDS = 1.0

# Default production base URL (configurable via env or parameter).
_DEFAULT_BASE_URL = (
    "https://prod-external-weather-api.birdseyeviewtechnologies.com/v1/in-depth"
)

# Peril sets by endpoint, using production enum names.
DAILY_PERILS: List[str] = ["Rain", "MaxWindSpeed", "MaxWindGust", "Lightning"]
EXPANDING_PERILS: List[str] = ["CumulativeRain"]


# ---------------------------------------------------------------------------
# Location helpers
# ---------------------------------------------------------------------------

def lookup_locations(
    mapping_df: pd.DataFrame,
    location_ids: Optional[List[int]] = None,
    countries: Optional[List[str]] = None,
    areas: Optional[List[str]] = None,
    cities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter the location mapping by optional criteria.

    All filters that are *not None* are applied (AND logic).  If every
    filter is *None* the full table is returned.
    """
    df = mapping_df.copy()
    if location_ids is not None:
        df = df[df["location_id"].isin(location_ids)]
    if countries is not None:
        df = df[df["country"].isin(countries)]
    if areas is not None:
        df = df[df["area"].isin(areas)]
    if cities is not None:
        df = df[df["city"].isin(cities)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------

def build_events_daily(
    locations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build a list of daily-endpoint event dicts from a locations frame.

    Returns
    -------
    (events, labels)
        *events* – list of dicts ready for the API payload.
        *labels* – parallel list of human-readable location descriptions.
    """
    events: List[Dict[str, Any]] = []
    labels: List[str] = []

    for idx, row in locations_df.iterrows():
        events.append({
            "index": int(idx),
            "location": "",
            "start_date": start_date,
            "end_date": end_date,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        })
        label_parts = [
            str(row.get("city", "")),
            str(row.get("area", "")),
            str(row.get("country", "")),
        ]
        label = ", ".join(p for p in label_parts if p)
        labels.append(f"{label} ({row['latitude']}, {row['longitude']})")

    return events, labels


def build_events_expanding(
    locations_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    start_hour: int = 0,
    end_hour: int = 23,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build a list of expanding-endpoint event dicts from a locations frame.

    Same as :func:`build_events_daily` but adds ``start_hour``,
    ``end_hour`` and ``tag`` fields required by the expanding endpoint.
    """
    events: List[Dict[str, Any]] = []
    labels: List[str] = []

    for idx, row in locations_df.iterrows():
        events.append({
            "index": int(idx),
            "tag": f"tag-{idx}",
            "location": "",
            "start_date": start_date,
            "end_date": end_date,
            "start_hour": start_hour,
            "end_hour": end_hour,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
        })
        label_parts = [
            str(row.get("city", "")),
            str(row.get("area", "")),
            str(row.get("country", "")),
        ]
        label = ", ".join(p for p in label_parts if p)
        labels.append(f"{label} ({row['latitude']}, {row['longitude']})")

    return events, labels


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------

def normalise_daily_results(
    raw_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten and normalise daily API responses into the CD rater schema.

    The production API returns probability values on a 0-100 scale
    (e.g. ``82``).  This function divides by 100 so the value stored in
    the CD rater schema is ``0.82``.

    Returns a list of dicts matching the ``adverse_weather_daily`` schema::

        {"peril": str, "index": int, "threshold": str, "value": float}
    """
    rows: List[Dict[str, Any]] = []
    for r in raw_results:
        idx = r.get("index")
        peril = r.get("peril")

        # Prod format: parallel threshold / probability arrays.
        thresholds = r.get("threshold") or []
        probabilities = r.get("probability") or []

        for t, p in zip(thresholds, probabilities):
            try:
                value = float(p) / 100.0
            except (TypeError, ValueError):
                value = None
            rows.append({
                "peril": peril,
                "index": idx,
                "threshold": str(t),
                "value": value,
            })

        # Legacy format: model dict mapping threshold -> value.
        if "model" in r and isinstance(r.get("model"), dict):
            for thresh, val in r["model"].items():
                try:
                    value = float(val) / 100.0
                except (TypeError, ValueError):
                    value = None
                rows.append({
                    "peril": peril,
                    "index": idx,
                    "threshold": str(thresh),
                    "value": value,
                })

    return rows


def normalise_expanding_results(
    raw_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten and normalise expanding API responses into the CD rater schema.

    Returns a list of dicts matching the ``adverse_weather_expanding`` schema::

        {"index": int, "window_index": int, "peril": str,
         "threshold": float, "value": float}

    Probability values are divided by 100 (same as daily).
    """
    rows: List[Dict[str, Any]] = []
    for r in raw_results:
        try:
            value = float(r.get("value", 0)) / 100.0
        except (TypeError, ValueError):
            value = None
        try:
            threshold = float(r.get("threshold", 0))
        except (TypeError, ValueError):
            threshold = None
        rows.append({
            "index": r.get("index"),
            "window_index": r.get("window_index"),
            "peril": r.get("peril"),
            "threshold": threshold,
            "value": value,
        })
    return rows


# ---------------------------------------------------------------------------
# Failed-event warning helpers
# ---------------------------------------------------------------------------

def build_failed_warnings(
    failed_events: List[Dict[str, Any]],
    labels: List[str],
    endpoint: str,
) -> List[Dict[str, str]]:
    """Build warning dicts for each failed event.

    Returns a list of ``{"label": ..., "message": ...}`` dicts suitable
    for appending to ``hxd.outputs.warnings``.
    """
    warnings: List[Dict[str, str]] = []
    for fe in failed_events:
        ev = fe["event"]
        idx = ev.get("index")
        lat = ev.get("latitude")
        lon = ev.get("longitude")
        label = labels[idx] if isinstance(idx, int) and idx < len(labels) else "N/A"
        warnings.append({
            "label": f"Adverse weather /{endpoint} failed",
            "message": (
                f"index={idx} lat={lat} lon={lon} label={label} — {fe['error']}"
            ),
        })
    return warnings


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def run_adverse_weather(
    locations_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    perils_daily: Optional[List[str]] = None,
    perils_expanding: Optional[List[str]] = None,
    window_days: int = 0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verify_ssl: bool = False,
    ca_bundle: Optional[str] = None,
    max_retries: int = 3,
    request_timeout: float = 300.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Run the adverse-weather API workflow for the CD rater.

    Returns
    -------
    (daily_rows, expanding_rows, failed_events, warnings)
        *daily_rows* matches the ``adverse_weather_daily`` schema.
        *expanding_rows* matches the ``adverse_weather_expanding`` schema.
        *failed_events* is a list of
        ``{"event": <dict>, "error": <str>}`` dicts.
        *warnings* is a list of ``{"label": ..., "message": ...}`` dicts
        ready for ``hxd.outputs.warnings``.
    """
    # Resolve defaults
    if api_key is None:
        api_key = os.environ.get("BEV_API_KEY_PROD") or os.environ.get("BEV_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key provided and BEV_API_KEY_PROD / BEV_API_KEY not set."
        )

    if base_url is None:
        base_url = os.environ.get("BEV_BASE_URL", _DEFAULT_BASE_URL)

    today = date.today().isoformat()
    start_date = start_date or today
    end_date = end_date or today

    if perils_daily is None:
        perils_daily = list(DAILY_PERILS)
    if perils_expanding is None:
        perils_expanding = list(EXPANDING_PERILS)

    all_failed: List[Dict[str, Any]] = []
    all_warnings: List[Dict[str, str]] = []
    daily_rows: List[Dict[str, Any]] = []
    expanding_rows: List[Dict[str, Any]] = []

    # --- Daily endpoint ------------------------------------------------
    if perils_daily:
        events_daily, labels_daily = build_events_daily(
            locations_df, start_date, end_date,
        )

        raw_daily, failed_daily = bev_task_batches_threaded(
            api_key=api_key,
            perils=perils_daily,
            endpoint="daily",
            event_set=events_daily,
            concurrency=_CD_CONCURRENCY,
            request_timeout=request_timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            ca_bundle=ca_bundle,
            base_url=base_url,
            max_combinations=ENDPOINT_MAX_COMBINATIONS["daily"],
            batch_pause_seconds=_CD_BATCH_PAUSE_SECONDS,
        )

        daily_rows = normalise_daily_results(raw_daily)
        all_failed.extend(failed_daily)
        all_warnings.extend(build_failed_warnings(failed_daily, labels_daily, "daily"))

    # --- Expanding endpoint --------------------------------------------
    if perils_expanding:
        events_expanding, labels_expanding = build_events_expanding(
            locations_df, start_date, end_date,
        )

        raw_expanding, failed_expanding = bev_task_batches_threaded(
            api_key=api_key,
            perils=perils_expanding,
            endpoint="expanding",
            event_set=events_expanding,
            window_days=window_days,
            concurrency=_CD_CONCURRENCY,
            request_timeout=request_timeout,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
            ca_bundle=ca_bundle,
            base_url=base_url,
            max_combinations=ENDPOINT_MAX_COMBINATIONS["expanding"],
            batch_pause_seconds=_CD_BATCH_PAUSE_SECONDS,
        )

        expanding_rows = normalise_expanding_results(raw_expanding)
        all_failed.extend(failed_expanding)
        all_warnings.extend(build_failed_warnings(failed_expanding, labels_expanding, "expanding"))

    return daily_rows, expanding_rows, all_failed, all_warnings
