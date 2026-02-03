"""BEV client helpers for batching and parallel requests.

Contains:
- bev_task_batches_threaded: parallel batched POSTs to BEV API
- batch_payload: split perils/events into batches
- _post_payload: helper to POST a single payload

This module is written to be easily unit-tested (Session is created inside function
so tests can monkeypatch requests.Session).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import datetime
from typing import List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _json_converter(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def batch_payload(perils: List[Any], events: List[Dict[str, Any]], window_days: Optional[int] = None, max_combinations: int = 4):
    """Batch perils and events into smaller payloads to respect API limits."""
    if not perils:
        return []
    max_events_per_batch = max(1, max_combinations // len(perils))

    if window_days is not None:
        return [
            {"perils": perils, "events": events[i:i + max_events_per_batch], "window_days": window_days}
            for i in range(0, len(events), max_events_per_batch)
        ]
    else:
        return [
            {"perils": perils, "events": events[i:i + max_events_per_batch]}
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


def bev_task_batches_threaded(api_key: str, perils: List[Any], endpoint: str, event_set: List[Dict[str, Any]], window_days: Optional[int] = None, concurrency: int = 30, request_timeout: float = 300.0, max_retries: int = 3, verify_ssl: bool = False, ca_bundle: Optional[str] = None, base_url: Optional[str] = None, endpoint_trailing: bool = False):
    """Call BEV weather API in parallel batches and return merged results.

    Args:
        api_key: API key for authentication.
        perils: Perils configuration; see endpoint schema.
        endpoint: 'daily' or 'expanding'.
        event_set: List of event dicts with 'index', 'location', 'start_date', 'end_date'.
        concurrency: Max concurrent requests.
        request_timeout: Per-request timeout in seconds.
        max_retries: Retry count for transient errors.
        verify_ssl: Whether to verify SSL certificates. Set to False for staging/non-prod APIs.
        ca_bundle: Optional path to a CA bundle file to use for verification (overrides verify_ssl when set).
        base_url: Optional base URL for the API (e.g., prod vs staging). If None, uses non-prod staging URL.
        endpoint_trailing: If True, include a trailing slash after the endpoint in the request URL.

    Returns:
        list: Combined list of API responses.
    """
    if endpoint not in ["daily", "expanding"]:
        raise ValueError(f"Invalid endpoint: {endpoint}. Must be 'daily' or 'expanding'")

    # Allow overriding the base URL (prod vs non-prod staging)
    if base_url is None:
        base_url = 'https://nonprodstage-weather-api-wrapper.birdseyeviewtechnologies.com/model/'
    # Construct URL; optionally include trailing slash depending on environment behavior
    if endpoint_trailing:
        url = base_url.rstrip('/') + '/' + endpoint + '/'
    else:
        url = base_url.rstrip('/') + '/' + endpoint

    json_data = {
        "perils": perils,
        "events": event_set
    }

    payloads = batch_payload(json_data['perils'], json_data['events'], window_days=window_days)

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'accept-encoding': 'gzip',
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
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [
            executor.submit(_post_payload, session, url, headers, pl, request_timeout)
            for pl in payloads
        ]
        for fut in as_completed(futures):
            r = fut.result()
            results.extend(r)

    return results


__all__ = [
    "bev_task_batches_threaded",
    "batch_payload",
    "_post_payload",
]
