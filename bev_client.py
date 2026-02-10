"""BEV client helpers for batching and parallel requests.

Contains:
- bev_task_batches_threaded: parallel batched POSTs to BEV API
- batch_payload: split perils/events into batches
- _post_payload: helper to POST a single payload

This module is written to be easily unit-tested (Session is created inside function
so tests can monkeypatch requests.Session).

Rate-limit defaults (event-peril combinations per request):
- /daily:     25
- /expanding: 10
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
import logging
import time
from typing import List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Default max event-peril combinations per request, by endpoint.
ENDPOINT_MAX_COMBINATIONS = {
    "daily": 25,
    "expanding": 10,
}


def _json_converter(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def batch_payload(
    perils: List[Any],
    events: List[Dict[str, Any]],
    window_days: Optional[int] = None,
    max_combinations: int = 4,
):
    """Batch perils and events into smaller payloads to respect API limits."""
    if not perils:
        return []
    # max_events_per_batch = max(1, max_combinations // len(perils))
    max_events_per_batch = max(1, max_combinations // len(perils))

    if window_days is not None:
        return [
            {
                "perils": perils,
                "events": events[i : i + max_events_per_batch],
                "window_days": window_days,
            }
            for i in range(0, len(events), max_events_per_batch)
        ]
    else:
        return [
            {"perils": perils, "events": events[i : i + max_events_per_batch]}
            for i in range(0, len(events), max_events_per_batch)
        ]


def _post_payload(
    session: requests.Session,
    url: str,
    headers: dict,
    payload: dict,
    timeout: float,
):
    """POST a single payload and return parsed JSON; raises for HTTP errors."""
    json_payload = json.dumps(payload, default=_json_converter)
    resp = session.post(url, headers=headers, data=json_payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()  # API returns a list per payload


def bev_task_batches_threaded(
    api_key: str,
    perils: List[Any],
    endpoint: str,
    event_set: List[Dict[str, Any]],
    window_days: Optional[int] = None,
    concurrency: int = 30,
    request_timeout: float = 300.0,
    max_retries: int = 3,
    verify_ssl: bool = False,
    ca_bundle: Optional[str] = None,
    base_url: Optional[str] = None,
    endpoint_trailing: bool = False,
    max_combinations: Optional[int] = None,
    batch_pause_seconds: float = 0,
):
    """Call BEV weather API in parallel batches and return merged results.

    Payloads are split so that no single request exceeds the provider's
    event-peril combination limit (25 for /daily, 10 for /expanding by
    default).  Payloads are sent in waves of ``concurrency`` requests;
    between waves the caller can inject a pause via ``batch_pause_seconds``
    to respect rate limits.

    Args:
        api_key: API key for authentication.
        perils: Perils configuration; see endpoint schema.
        endpoint: 'daily' or 'expanding'.
        event_set: List of event dicts with keys like 'index', 'location',
            'latitude', 'longitude', 'start_date', 'end_date'.
        window_days: For 'expanding' endpoint only.
        concurrency: Max concurrent requests per wave.
        request_timeout: Per-request timeout in seconds.
        max_retries: Retry count for transient errors.
        verify_ssl: Whether to verify SSL certificates.
        ca_bundle: Optional path to a CA bundle file (overrides verify_ssl).
        base_url: Optional base URL for the API. Defaults to staging URL.
        endpoint_trailing: If True, append trailing slash to endpoint URL.
        max_combinations: Max event-peril combinations per request.
            Defaults to 25 for /daily and 10 for /expanding.
        batch_pause_seconds: Seconds to sleep between concurrency waves.
            Set to 0 (default) to disable pausing.

    Returns:
        tuple: ``(results, failed_events)`` where *results* is the combined
        list of API responses and *failed_events* is a list of dicts
        ``{"event": <event_dict>, "error": <str>}`` for locations that
        returned 500 errors even after individual retries.
    """
    if endpoint not in ["daily", "expanding"]:
        raise ValueError(
            f"Invalid endpoint: {endpoint}. Must be 'daily' or 'expanding'"
        )

    # Derive default max_combinations from endpoint when not explicitly set
    if max_combinations is None:
        max_combinations = ENDPOINT_MAX_COMBINATIONS.get(endpoint, 25)

    # Allow overriding the base URL (prod vs non-prod staging)
    if base_url is None:
        base_url = "https://nonprodstage-weather-api-wrapper.birdseyeviewtechnologies.com/model/"
    # Construct URL; optionally include trailing slash depending on environment behavior
    if endpoint_trailing:
        url = base_url.rstrip("/") + "/" + endpoint + "/"
    else:
        url = base_url.rstrip("/") + "/" + endpoint

    json_data = {"perils": perils, "events": event_set}

    payloads = batch_payload(
        json_data["perils"],
        json_data["events"],
        window_days=window_days,
        max_combinations=max_combinations,
    )

    logger.info(
        f"Batched {len(event_set)} events x {len(perils)} perils into "
        f"{len(payloads)} payloads (max_combinations={max_combinations})"
    )

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "accept-encoding": "gzip",
    }

    retries = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    adapter = HTTPAdapter(
        pool_connections=max(1, concurrency),
        pool_maxsize=max(1, concurrency),
        max_retries=retries,
    )
    session = requests.Session()
    session.headers.update(headers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Support passing a CA bundle path (string) or a boolean verify flag
    if ca_bundle is not None:
        session.verify = ca_bundle
    else:
        session.verify = verify_ssl

    results = []
    failed_events = []

    # Send payloads in waves of `concurrency`, pausing between waves.
    wave_size = max(1, concurrency)
    for wave_start in range(0, len(payloads), wave_size):
        wave = payloads[wave_start : wave_start + wave_size]

        with ThreadPoolExecutor(max_workers=wave_size) as executor:
            future_to_payload = {
                executor.submit(
                    _post_payload, session, url, headers, pl, request_timeout
                ): pl
                for pl in wave
            }
            for fut in as_completed(future_to_payload):
                try:
                    r = fut.result()
                    results.extend(r)
                except Exception as exc:
                    # Batch failed – retry each event individually to isolate
                    # the problematic location(s).
                    failed_payload = future_to_payload[fut]
                    batch_events = failed_payload["events"]
                    logger.warning(
                        f"Batch of {len(batch_events)} events failed: {exc}. "
                        f"Retrying events one-by-one to isolate failures..."
                    )
                    for event in batch_events:
                        single_payload = {
                            "perils": failed_payload["perils"],
                            "events": [event],
                        }
                        if "window_days" in failed_payload:
                            single_payload["window_days"] = failed_payload[
                                "window_days"
                            ]
                        try:
                            r = _post_payload(
                                session, url, headers, single_payload,
                                request_timeout,
                            )
                            results.extend(r)
                            logger.info(
                                f"  Individual retry succeeded: index={event.get('index')} "
                                f"lat={event.get('latitude')} lon={event.get('longitude')} "
                                f"location={event.get('location', '')!r}"
                            )
                        except Exception as single_exc:
                            failed_events.append({
                                "event": event,
                                "error": str(single_exc),
                            })
                            logger.error(
                                f"  FAILED LOCATION: index={event.get('index')} "
                                f"lat={event.get('latitude')} lon={event.get('longitude')} "
                                f"location={event.get('location', '')!r} — {single_exc}"
                            )

        # Pause between waves (skip after the last wave)
        if batch_pause_seconds > 0 and wave_start + wave_size < len(payloads):
            logger.info(
                f"Pausing {batch_pause_seconds}s between payload waves "
                f"(completed {wave_start + len(wave)}/{len(payloads)})"
            )
            time.sleep(batch_pause_seconds)

    if failed_events:
        logger.warning(
            f"{len(failed_events)} event(s) failed after individual retries. "
            f"Returning {len(results)} successful results."
        )

    return results, failed_events


__all__ = [
    "bev_task_batches_threaded",
    "batch_payload",
    "_post_payload",
]
