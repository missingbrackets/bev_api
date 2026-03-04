"""Lightweight BEV API client for the Streamlit app."""

import datetime
import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DEFAULT_PROD_BASE = (
    "https://prod-external-weather-api.birdseyeviewtechnologies.com/v1/in-depth"
)

ENDPOINT_MAX_COMBINATIONS = {"daily": 25, "expanding": 10}


def _json_converter(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _post_payload(session, url, headers, payload, timeout):
    json_payload = json.dumps(payload, default=_json_converter)
    resp = session.post(url, headers=headers, data=json_payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def call_bev_api(
    api_key: str,
    perils: List[str],
    events: List[Dict[str, Any]],
    endpoint: str = "daily",
    base_url: str = DEFAULT_PROD_BASE,
    max_combinations: Optional[int] = None,
    request_timeout: float = 300.0,
    verify_ssl: bool = False,
) -> List[Dict[str, Any]]:
    """Call the BEV API and return a flat list of result dicts.

    This is a simplified single-threaded version suitable for small
    numbers of events from the Streamlit UI.
    """
    if max_combinations is None:
        max_combinations = ENDPOINT_MAX_COMBINATIONS.get(endpoint, 25)

    url = base_url.rstrip("/") + "/" + endpoint

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "accept-encoding": "gzip",
    }

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = verify_ssl

    # Batch events to stay under max_combinations
    max_per_batch = max(1, max_combinations // len(perils))
    payloads = [
        {"perils": perils, "events": events[i : i + max_per_batch]}
        for i in range(0, len(events), max_per_batch)
    ]

    results: List[Dict[str, Any]] = []
    for payload in payloads:
        r = _post_payload(session, url, headers, payload, request_timeout)
        results.extend(r)

    return results
