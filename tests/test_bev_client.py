import os
import json
import pytest
import logging
import time

from bev_client import batch_payload, bev_task_batches_threaded

logger = logging.getLogger(__name__)


def test_batch_payload_basic():
    logger.info("=== test_batch_payload_basic ===")
    perils = ["rain", "wind"]
    events = [
        {"index": i, "location": f"L{i}", "start_date": "2025-07-14", "end_date": "2025-07-14"}
        for i in range(6)
    ]
    logger.info(f"Input: {len(perils)} perils, {len(events)} events")

    batches = batch_payload(perils, events, window_days=None, max_combinations=4)
    # With 2 perils and max_combinations=4 -> max_events_per_batch = 2
    logger.info(f"Output: {len(batches)} batches")
    for i, batch in enumerate(batches):
        logger.info(f"  Batch {i}: {len(batch['events'])} events")
    
    assert all("perils" in b and "events" in b for b in batches)
    assert sum(len(b["events"]) for b in batches) == len(events)
    logger.info("✓ Batching test passed")


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeSession:
    def __init__(self, responses=None):
        self._responses = responses or []
        self.headers = {}
        self.post_count = 0

    def headers_update(self, d):
        self.headers.update(d)

    def mount(self, prefix, adapter):
        logger.info(f"  [MOUNT] {prefix} -> {adapter.__class__.__name__}")

    def post(self, url, headers=None, data=None, timeout=None):
        self.post_count += 1
        logger.info(f"  [POST #{self.post_count}] URL: {url}")
        logger.debug(f"  [POST] Headers: {headers}")
        if data:
            logger.debug(f"  [POST] Payload: {data[:200]}...")  # First 200 chars
        logger.debug(f"  [POST] Timeout: {timeout}s")
        
        # Return a realistic API response
        fake_response = [
            {
                "index": 0,
                "peril": "rain",
                "model": {
                    "threshold_10": 0.15,
                    "threshold_50": 0.45,
                    "threshold_90": 0.82,
                }
            }
        ]
        logger.info(f"  [RESPONSE #{self.post_count}] Returning {len(fake_response)} records")
        return _FakeResponse(fake_response)


def test_bev_task_batches_threaded_mock(monkeypatch):
    logger.info("=== test_bev_task_batches_threaded_mock ===")
    # Replace requests.Session with one that returns predictable responses
    monkeypatch.setattr("bev_client.requests.Session", lambda: _FakeSession())

    perils = ["rain"]
    events = [
        {"index": i, "location": f"L{i}", "start_date": "2025-07-14", "end_date": "2025-07-14"}
        for i in range(3)
    ]
    
    logger.info(f"Calling bev_task_batches_threaded with {len(events)} events, concurrency=2")
    start = time.time()
    result = bev_task_batches_threaded(api_key="dummy", perils=perils, endpoint="daily", event_set=events, concurrency=2)
    elapsed = time.time() - start
    
    logger.info(f"✓ API call completed in {elapsed:.2f}s")
    logger.info(f"Total results: {len(result)} records")
    logger.info(f"Result structure: {json.dumps(result[:2], indent=2)}")
    
    assert isinstance(result, list), "Result should be a list"
    assert len(result) > 0, "Result should not be empty"
    assert all("index" in r and "peril" in r and "model" in r for r in result), "Each record should have index/peril/model"
    logger.info("✓ All assertions passed")


@pytest.mark.skipif(not os.environ.get("BEV_API_KEY"), reason="Requires BEV_API_KEY for integration test")
def test_bev_task_batches_integration():
    logger.info("=== test_bev_task_batches_integration (REAL API) ===")
    api_key = os.environ.get("BEV_API_KEY")
    perils = ["Rain"]  # MUST be capitalized: 'Rain', 'Max Wind Gust', 'Max Wind Speed', or 'Lightning'
    events = [
        {"index": 0, "location": "Dublin", "start_date": "2025-07-14", "end_date": "2025-07-14"}
    ]

    logger.info(f"Connecting to real API with api_key={api_key[:10]}...")
    logger.info(f"Payload: perils={perils}, events={events}")
    
    start = time.time()
    try:
        result = bev_task_batches_threaded(api_key=api_key, perils=perils, endpoint="daily", event_set=events, concurrency=1, verify_ssl=False)
        elapsed = time.time() - start
        
        logger.info(f"✓ Real API call completed in {elapsed:.2f}s")
        logger.info(f"Total results: {len(result)} records")
        if result:
            logger.info(f"First result: {json.dumps(result[0], indent=2)}")
        
        assert isinstance(result, list)
        if result:
            assert all("index" in r and "peril" in r and "model" in r for r in result)
        logger.info("✓ Integration test passed")
    except Exception as exc:
        logger.error(f"✗ API call failed: {exc}")
        logger.error(f"Exception type: {type(exc).__name__}")
        if hasattr(exc, 'response'):
            logger.error(f"Response status: {exc.response.status_code}")
            logger.error(f"Response body: {exc.response.text}")
        raise
