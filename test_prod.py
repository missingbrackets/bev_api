"""Test BEV Prod API only.

Requests use latitude/longitude coordinates.  The ``location`` field in the
API payload is left blank; a human-friendly label is kept for logging and
summary output only.

Usage:
    # Random 5 locations (lat/long from all_lat_longs.csv)
    python test_prod.py --num-locations 5 --perils Rain,MaxWindSpeed,Lightning

    # From input CSV (must include latitude, longitude columns)
    python test_prod.py --input-csv 02_Data/test_events.csv

    # Override rate-limit batching and pause between waves
    python test_prod.py --num-locations 20 --max-combinations 10 --batch-pause-seconds 2

    # Reproducible run with a fixed seed
    python test_prod.py --num-locations 10 --random-seed 42

Environment variables expected:
- BEV_API_KEY_PROD: prod API key (required, falls back to BEV_API_KEY)

Outputs (in output dir):
- bev_prod_<timestamp>.json (raw API response)
- bev_prod_<timestamp>.xlsx (with sheets: results, summary)

Default locations file: 02_Data/all_lat_longs.csv
    Columns: country, area, city, location_id, latitude, longitude

"""
from pathlib import Path
import argparse
import json
import os
import time
import logging
from typing import List, Dict, Any, Tuple
from datetime import date

import numpy as np
import pandas as pd

from bev_client import bev_task_batches_threaded

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_PROD_BASE = 'https://prod-external-weather-api.birdseyeviewtechnologies.com/v1/in-depth'


def flatten_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert API results into long DataFrame handling the prod response shape.

    Expected shape:
    - {index, peril, latitude, longitude, threshold: [...], probability: [...], unit}
    """
    rows = []
    for r in results:
        idx = r.get('index')
        peril = r.get('peril')
        lat = r.get('latitude')
        lon = r.get('longitude')
        unit = r.get('unit')

        # Prod response: parallel arrays 'threshold' and 'probability'
        if 'threshold' in r and 'probability' in r:
            thresh_list = r.get('threshold') or []
            prob_list = r.get('probability') or []
            for t, p in zip(thresh_list, prob_list):
                try:
                    pf = float(p)
                except Exception:
                    pf = None
                rows.append({
                    'index': idx,
                    'peril': peril,
                    'threshold': str(t),
                    'probability': pf,
                    'latitude': lat,
                    'longitude': lon,
                    'unit': unit,
                })
            continue

        # Legacy response: 'model' is a dict mapping thresholds -> value
        if 'model' in r and isinstance(r.get('model'), dict):
            for thresh, val in r['model'].items():
                try:
                    valf = float(val)
                except Exception:
                    valf = None
                rows.append({
                    'index': idx,
                    'peril': peril,
                    'threshold': str(thresh),
                    'probability': valf,
                    'latitude': lat,
                    'longitude': lon,
                    'unit': unit,
                })
            continue

    return pd.DataFrame(rows)


def run_prod_api(
    api_key: str,
    perils: List[str],
    endpoint: str,
    events: List[Dict[str, Any]],
    base_url: str,
    verify_ssl: bool,
    ca_bundle: str = None,
    window_days: int = 0,
    concurrency: int = 10,
    max_combinations: int = None,
    batch_pause_seconds: float = 0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], float]:
    """Run Prod API and return (dataframe, raw_results, elapsed_seconds)."""
    start = time.time()
    endpoint_trailing = base_url.endswith('/')
    full_url = base_url.rstrip('/') + '/' + endpoint + ('/' if endpoint_trailing else '')
    logger.info(f"Running prod ({base_url}) -> {full_url} for {len(events)} events and perils={perils}")

    try:
        bev_kwargs = dict(
            api_key=api_key,
            perils=perils,
            endpoint=endpoint,
            event_set=events,
            concurrency=concurrency,
            request_timeout=300.0,
            max_retries=3,
            verify_ssl=verify_ssl,
            ca_bundle=ca_bundle,
            base_url=base_url,
            endpoint_trailing=endpoint_trailing,
            max_combinations=max_combinations,
            batch_pause_seconds=batch_pause_seconds,
        )

        if endpoint == "expanding":
            bev_kwargs["window_days"] = window_days

        results = bev_task_batches_threaded(**bev_kwargs)
        logger.info(f"Prod call succeeded with base {base_url}")
    except Exception as exc:
        logger.error(f"Prod call to {full_url} failed: {exc}")
        try:
            import requests
            if isinstance(exc, requests.exceptions.HTTPError) and hasattr(exc, 'response') and exc.response is not None:
                logger.error(f"Response status: {exc.response.status_code}")
                logger.error(f"Response body: {exc.response.text}")
        except Exception:
            pass
        raise

    elapsed = time.time() - start
    logger.info(f"Prod run finished in {elapsed:.2f}s; got {len(results)} records")

    df = flatten_results(results)
    return df, results, elapsed


def load_events_from_csv(csv_path: str, endpoint: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load events from a CSV file.

    The API payload uses latitude/longitude; the ``location`` field is sent
    as an empty string.  A human-friendly label is returned separately for
    logging purposes.

    Expected columns:
    - index: unique identifier for the event
    - start_date, end_date: YYYY-MM-DD
    - latitude, longitude: numeric coordinates

    Optional columns:
    - location: kept only as a display label (not sent to API)
    - start_hour, end_hour: for expanding endpoint (default 0/23)
    - tag: for expanding endpoint (auto-generated if absent)

    Returns:
        (events, labels) where *events* are the API payloads and *labels*
        are human-readable location strings for logging.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ['index', 'start_date', 'end_date', 'latitude', 'longitude']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Convert dates to string format if they're datetime
    df['start_date'] = pd.to_datetime(df['start_date']).dt.strftime('%Y-%m-%d')
    df['end_date'] = pd.to_datetime(df['end_date']).dt.strftime('%Y-%m-%d')

    # Build display labels from location column or lat/long
    if 'location' in df.columns:
        labels = [
            f"{row.get('location', '')} ({row['latitude']}, {row['longitude']})"
            for _, row in df.iterrows()
        ]
    else:
        labels = [
            f"({row['latitude']}, {row['longitude']})"
            for _, row in df.iterrows()
        ]

    if endpoint == 'daily':
        df['location'] = ''
        events = df[['index', 'location', 'start_date', 'end_date',
                      'latitude', 'longitude']].to_dict(orient='records')
    else:
        # expanding endpoint: add optional fields with defaults
        df['location'] = ''
        df['start_hour'] = df.get('start_hour', 0).fillna(0).astype(int)
        df['end_hour'] = df.get('end_hour', 23).fillna(23).astype(int)
        if 'tag' not in df.columns:
            df['tag'] = [f"tag-{i}" for i in df['index']]

        events = df[['index', 'tag', 'location', 'start_date', 'end_date',
                     'start_hour', 'end_hour', 'latitude', 'longitude']].to_dict(orient='records')

    return events, labels


def build_random_events(
    locations_file: str,
    num_locations: int,
    endpoint: str,
    random_seed: int = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build random events by sampling lat/long locations.

    Reads ``locations_file`` (default: ``02_Data/all_lat_longs.csv``) which
    must contain columns: country, area, city, latitude, longitude.

    Sampling is truly random each run unless ``random_seed`` is provided.

    The API payload sets ``location`` to an empty string and uses
    ``latitude``/``longitude`` instead.  A human-friendly label list is
    returned alongside the events for logging.

    Returns:
        (events, labels) – events for the API and labels for display.
    """
    loc_df = pd.read_csv(locations_file)
    loc_df.columns = [c.strip().lower() for c in loc_df.columns]

    required = ['country', 'area', 'city', 'latitude', 'longitude']
    missing = [c for c in required if c not in loc_df.columns]
    if missing:
        raise ValueError(f"Locations file missing required columns: {missing}")

    loc_df = loc_df.dropna(subset=['country', 'area', 'city', 'latitude', 'longitude'])

    n = min(num_locations, len(loc_df))
    sampled = loc_df.sample(n=n, random_state=random_seed)
    sampled = sampled.reset_index(drop=True)

    # Build event info
    df_event_info = sampled.copy()
    df_event_info['index'] = df_event_info.index

    # Human-readable label for logging only
    df_event_info['_label'] = (
        df_event_info['city'].astype(str).str.strip() + ", " +
        df_event_info['area'].astype(str).str.strip() + ", " +
        df_event_info['country'].astype(str).str.strip()
    )

    labels = [
        f"{row['_label']} ({row['latitude']}, {row['longitude']})"
        for _, row in df_event_info.iterrows()
    ]

    ev_date = date.today().isoformat()
    df_event_info['start_date'] = ev_date
    df_event_info['end_date'] = ev_date

    # API payload: location is blank, use lat/long
    df_event_info['location'] = ''

    if endpoint == 'daily':
        events = df_event_info[['index', 'location', 'start_date', 'end_date',
                                 'latitude', 'longitude']].to_dict(orient='records')
    else:
        # expanding endpoint
        df_event_info['start_hour'] = 0
        df_event_info['end_hour'] = 23
        df_event_info['tag'] = [f"tag-{i}" for i in df_event_info['index']]
        events = df_event_info[['index', 'tag', 'location', 'start_date', 'end_date',
                                 'start_hour', 'end_hour', 'latitude', 'longitude']].to_dict(orient='records')

    return events, labels


def build_summary_df(
    events: List[Dict[str, Any]],
    labels: List[str],
    perils: List[str],
    elapsed: float
) -> pd.DataFrame:
    """Build summary DataFrame with timing and location/peril info.

    ``labels`` is a parallel list of human-readable location descriptions
    (same length as ``events``) used for the summary sheet.
    """

    rows = [
        {'metric': 'elapsed_seconds', 'value': f"{elapsed:.2f}"},
        {'metric': '', 'value': ''},
        {'metric': 'perils', 'value': ', '.join(perils)},
        {'metric': 'num_perils', 'value': str(len(perils))},
        {'metric': '', 'value': ''},
        {'metric': 'num_locations', 'value': str(len(events))},
        {'metric': '', 'value': ''},
        {'metric': '--- Locations ---', 'value': ''},
    ]

    for i, event in enumerate(events):
        idx = event.get('index', 'N/A')
        label = labels[i] if i < len(labels) else 'N/A'
        start_date = event.get('start_date', 'N/A')
        end_date = event.get('end_date', 'N/A')
        rows.append({
            'metric': f'index_{idx}',
            'value': f"{label} ({start_date} to {end_date})"
        })

    return pd.DataFrame(rows)


def write_excel_output(
    df_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    ts: int
) -> Path:
    """Write results and summary to Excel file."""

    excel_path = output_dir / f"bev_prod_{ts}.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='results', index=False)
        summary_df.to_excel(writer, sheet_name='summary', index=False)

    logger.info(f"Saved Excel output -> {excel_path}")
    return excel_path


def main():
    parser = argparse.ArgumentParser(
        description='Test BEV Prod API using latitude/longitude coordinates.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Random 5 locations (truly random each run)\n"
            "  python test_prod.py --num-locations 5\n\n"
            "  # Reproducible run with seed\n"
            "  python test_prod.py --num-locations 10 --random-seed 42\n\n"
            "  # From CSV with lat/long columns\n"
            "  python test_prod.py --input-csv 02_Data/test_events.csv\n\n"
            "  # Custom batching for rate limits\n"
            "  python test_prod.py --num-locations 50 --max-combinations 10 "
            "--batch-pause-seconds 2\n"
        ),
    )
    parser.add_argument('--input-csv', type=str, default=None,
                        help='CSV file with events (must include latitude, longitude columns)')
    parser.add_argument('--locations-file', type=str,
                        default='02_Data/all_lat_longs.csv',
                        help='CSV with columns: country, area, city, latitude, longitude '
                             '(default: 02_Data/all_lat_longs.csv)')
    parser.add_argument('--num-locations', type=int, default=5,
                        help='Number of random locations to sample (default: 5)')
    parser.add_argument('--perils', type=str, default='Rain',
                        help='Comma-separated list of perils in prod format (default: Rain)')
    parser.add_argument('--window-days', type=int, default=0,
                        help='window_days for expanding endpoint (default: 0)')
    parser.add_argument('--output-dir', type=str, default='03_Analysis',
                        help='Directory to save outputs (default: 03_Analysis)')
    parser.add_argument('--endpoint', type=str, choices=['daily', 'expanding'], default='daily',
                        help="API endpoint: 'daily' or 'expanding' (default: daily)")
    parser.add_argument('--verify-ssl', action='store_true',
                        help='Verify SSL for prod')
    parser.add_argument('--ca-bundle', type=str, default=None,
                        help='Path to CA bundle file')
    parser.add_argument('--prod-base', type=str, default=None,
                        help='Override prod base URL')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for location sampling (omit for truly random)')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Number of concurrent requests per wave (default: 10)')
    parser.add_argument('--max-combinations', type=int, default=None,
                        help='Max event-peril combinations per request. '
                             'Defaults to 25 for /daily and 10 for /expanding.')
    parser.add_argument('--batch-pause-seconds', type=float, default=1.0,
                        help='Seconds to pause between concurrency waves (default: 1.0)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse perils
    perils = [p.strip() for p in args.perils.split(',') if p.strip()]
    if not perils:
        perils = ['Rain']

    # Build events from CSV or random sampling
    if args.input_csv:
        logger.info(f"Loading events from CSV: {args.input_csv}")
        events, labels = load_events_from_csv(args.input_csv, args.endpoint)
    else:
        logger.info(f"Sampling {args.num_locations} random locations from {args.locations_file}")
        events, labels = build_random_events(
            args.locations_file,
            args.num_locations,
            args.endpoint,
            args.random_seed
        )

    # Get API key
    api_key = os.environ.get('BEV_API_KEY_PROD') or os.environ.get('BEV_API_KEY')
    if not api_key:
        raise RuntimeError('Prod API key not set in BEV_API_KEY_PROD or BEV_API_KEY')

    prod_base = args.prod_base or DEFAULT_PROD_BASE

    # Log test configuration before running
    logger.info("=" * 60)
    logger.info("TEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Window days: {args.window_days}")
    logger.info(f"Max combinations: {args.max_combinations or '(endpoint default)'}")
    logger.info(f"Batch pause: {args.batch_pause_seconds}s")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info("")
    logger.info(f"PERILS: {perils}")
    logger.info("")
    logger.info(f"LOCATIONS ({len(events)} total):")
    for i, event in enumerate(events):
        idx = event.get('index', 'N/A')
        label = labels[i] if i < len(labels) else 'N/A'
        start_date = event.get('start_date', 'N/A')
        end_date = event.get('end_date', 'N/A')
        logger.info(f"  [{idx}] {label} ({start_date} to {end_date})")
    logger.info("=" * 60)
    logger.info("")

    ts = int(time.time())

    # Run prod API
    df_results, results, elapsed = run_prod_api(
        api_key=api_key,
        perils=perils,
        endpoint=args.endpoint,
        events=events,
        base_url=prod_base,
        verify_ssl=args.verify_ssl,
        ca_bundle=args.ca_bundle,
        window_days=args.window_days,
        concurrency=args.concurrency,
        max_combinations=args.max_combinations,
        batch_pause_seconds=args.batch_pause_seconds,
    )

    # Save raw JSON
    json_path = output_dir / f"bev_prod_{ts}.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved JSON -> {json_path}")

    # Build summary
    summary_df = build_summary_df(events, labels, perils, elapsed)

    # Write Excel output
    excel_path = write_excel_output(df_results, summary_df, output_dir, ts)

    logger.info('Test complete')

    # Print quick results summary
    if not df_results.empty:
        logger.info(f"Results: {len(df_results)} rows")
        logger.info('\n' + df_results.head(20).to_string(index=False))
    else:
        logger.info('No results returned')


if __name__ == '__main__':
    main()
