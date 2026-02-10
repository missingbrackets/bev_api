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

import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bev_client import bev_task_batches_threaded, ENDPOINT_MAX_COMBINATIONS

logger = logging.getLogger(__name__)

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

# Path to the *new* location mapping that includes latitude / longitude.
_PARAMETER_TABLES_DIR = Path(__file__).resolve().parent.parent.parent / "parameter_tables"
_LOCATION_MAPPING_CSV = _PARAMETER_TABLES_DIR / "area_city_location_id_mapping_new.csv"

# Columns the new mapping file is expected to contain.
_REQUIRED_MAPPING_COLS = {"country", "area", "city", "location_id", "latitude", "longitude"}


# ---------------------------------------------------------------------------
# Location table helpers
# ---------------------------------------------------------------------------

def load_location_mapping(
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load the area/city/location-id mapping that includes lat/lon.

    Parameters
    ----------
    csv_path : str, optional
        Override the default CSV path.  When *None* the bundled
        ``parameter_tables/area_city_location_id_mapping_new.csv`` is used.

    Returns
    -------
    pd.DataFrame
        Columns: country, area, city, location_id, latitude, longitude.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(csv_path) if csv_path else _LOCATION_MAPPING_CSV
    if not path.exists():
        raise FileNotFoundError(f"Location mapping CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = _REQUIRED_MAPPING_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Location mapping CSV is missing columns: {sorted(missing)}"
        )

    # Ensure numeric coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    return df


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

    Parameters
    ----------
    locations_df : pd.DataFrame
        Must contain ``latitude``, ``longitude`` (and ideally ``city``,
        ``area``, ``country`` for labels).
    start_date, end_date : str
        ISO-format date strings (YYYY-MM-DD).

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

    Expected input shape per result dict::

        {
            "index": int,
            "peril": str,
            "threshold": [str, ...],
            "probability": [float, ...],
            "unit": str,
            ...
        }

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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the adverse-weather API workflow for the CD rater.

    This is the **primary entry point** called by the CD rater task
    pipeline.  It replicates the BEV batching / error-isolation
    methodology with CD-rater-specific settings:

    * concurrency = 1 (single-threaded)
    * 1 s pause after every payload (wave size = 1)
    * Automatic per-event retry on batch 500 errors
    * Probability values normalised (÷ 100)

    Parameters
    ----------
    locations_df : pd.DataFrame
        Locations to query.  Must contain ``latitude``, ``longitude``.
    start_date, end_date : str, optional
        ISO date strings.  Default to today.
    perils_daily : list[str], optional
        Perils for the ``/daily`` endpoint.  Defaults to
        ``["Rain", "MaxWindSpeed", "MaxWindGust", "Lightning"]``.
    perils_expanding : list[str], optional
        Perils for the ``/expanding`` endpoint.  Defaults to
        ``["CumulativeRain"]``.  Pass an empty list to skip expanding.
    window_days : int
        Window days for the expanding endpoint.
    api_key : str, optional
        BEV API key.  Falls back to ``BEV_API_KEY_PROD`` /
        ``BEV_API_KEY`` environment variables.
    base_url : str, optional
        API base URL.  Defaults to production.
    verify_ssl : bool
        Whether to verify TLS certificates.
    ca_bundle : str, optional
        Path to a CA bundle (overrides *verify_ssl*).
    max_retries : int
        Retry count for transient HTTP errors.
    request_timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    (daily_rows, expanding_rows, failed_events)
        *daily_rows* matches the ``adverse_weather_daily`` schema.
        *expanding_rows* matches the ``adverse_weather_expanding`` schema.
        *failed_events* is a list of
        ``{"event": <dict>, "error": <str>}`` dicts.
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
    daily_rows: List[Dict[str, Any]] = []
    expanding_rows: List[Dict[str, Any]] = []

    # --- Daily endpoint ------------------------------------------------
    if perils_daily:
        events_daily, labels_daily = build_events_daily(
            locations_df, start_date, end_date,
        )
        logger.info(
            "CD rater adverse-weather: calling /daily for %d locations, "
            "perils=%s", len(events_daily), perils_daily,
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

        _log_failed(failed_daily, labels_daily, "daily")

    # --- Expanding endpoint --------------------------------------------
    if perils_expanding:
        events_expanding, labels_expanding = build_events_expanding(
            locations_df, start_date, end_date,
        )
        logger.info(
            "CD rater adverse-weather: calling /expanding for %d locations, "
            "perils=%s, window_days=%d",
            len(events_expanding), perils_expanding, window_days,
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

        _log_failed(failed_expanding, labels_expanding, "expanding")

    logger.info(
        "CD rater adverse-weather complete: %d daily rows, %d expanding rows, "
        "%d failed events.",
        len(daily_rows), len(expanding_rows), len(all_failed),
    )

    return daily_rows, expanding_rows, all_failed


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_failed(
    failed_events: List[Dict[str, Any]],
    labels: List[str],
    endpoint: str,
) -> None:
    """Emit warning-level log lines for each failed event."""
    if not failed_events:
        return
    logger.warning(
        "%d event(s) failed on /%s endpoint:", len(failed_events), endpoint,
    )
    for fe in failed_events:
        ev = fe["event"]
        idx = ev.get("index")
        lat = ev.get("latitude")
        lon = ev.get("longitude")
        label = labels[idx] if isinstance(idx, int) and idx < len(labels) else "N/A"
        logger.warning(
            "  FAILED index=%s lat=%s lon=%s label=%r — %s",
            idx, lat, lon, label, fe["error"],
        )
