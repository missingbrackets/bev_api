"""Test BEV Prod API only.

Usage:
    # Random locations with specified perils
    python test_prod.py --num-locations 5 --perils Rain,MaxWindSpeed,Lightning

    # From input CSV
    python test_prod.py --input-csv 02_Data/test_events.csv

    # From input CSV with specific perils
    python test_prod.py --input-csv 02_Data/test_events.csv --perils Rain,MaxWindSpeed

Environment variables expected:
- BEV_API_KEY_PROD: prod API key (required)

Outputs (in output dir):
- bev_prod_<timestamp>.json (raw API response)
- bev_prod_<timestamp>.xlsx (with sheets: results, summary)

Input CSV format (see 02_Data/example_test_events.csv):
    index,location,start_date,end_date
    0,"New York, New York, United States",2026-02-01,2026-02-01
    1,"London, England, United Kingdom",2026-02-01,2026-02-01

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
    window_days: int = 0
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
            concurrency=10,
            request_timeout=300.0,
            max_retries=3,
            verify_ssl=verify_ssl,
            ca_bundle=ca_bundle,
            base_url=base_url,
            endpoint_trailing=endpoint_trailing,
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


def load_events_from_csv(csv_path: str, endpoint: str) -> List[Dict[str, Any]]:
    """Load events from a CSV file.

    Expected columns:
    - index: unique identifier for the event
    - location: location string (e.g., "City, Area, Country")
    - start_date: start date (YYYY-MM-DD)
    - end_date: end date (YYYY-MM-DD)

    Optional columns for 'expanding' endpoint:
    - start_hour: (default 0)
    - end_hour: (default 23)
    - latitude: (default 0)
    - longitude: (default 0)
    - tag: (default generated from location)
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ['index', 'location', 'start_date', 'end_date']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Convert dates to string format if they're datetime
    df['start_date'] = pd.to_datetime(df['start_date']).dt.strftime('%Y-%m-%d')
    df['end_date'] = pd.to_datetime(df['end_date']).dt.strftime('%Y-%m-%d')

    if endpoint == 'daily':
        events = df[['index', 'location', 'start_date', 'end_date']].to_dict(orient='records')
    else:
        # expanding endpoint: add optional fields with defaults
        df['start_hour'] = df.get('start_hour', 0).fillna(0).astype(int)
        df['end_hour'] = df.get('end_hour', 23).fillna(23).astype(int)
        df['latitude'] = df.get('latitude', 0).fillna(0)
        df['longitude'] = df.get('longitude', 0).fillna(0)
        if 'tag' not in df.columns:
            df['tag'] = df['location'].apply(lambda x: f"tag-{str(x)[:20]}")

        events = df[['index', 'tag', 'location', 'start_date', 'end_date',
                     'start_hour', 'end_hour', 'latitude', 'longitude']].to_dict(orient='records')

    return events


def build_random_events(
    locations_file: str,
    num_locations: int,
    endpoint: str,
    random_seed: int = None
) -> List[Dict[str, Any]]:
    """Build random events from the locations mapping file."""
    loc_df = pd.read_csv(locations_file, usecols=[0, 1, 2], names=['country', 'area', 'city'], header=0)
    loc_df = loc_df.dropna(subset=['country', 'area', 'city'])

    n = min(num_locations, len(loc_df))
    sampled = loc_df.sample(n=n, random_state=random_seed)
    sampled = sampled.reset_index(drop=True)

    # Build event info
    df_event_info = sampled.copy()
    df_event_info['index'] = df_event_info.index
    df_event_info['location'] = (
        df_event_info['city'].astype(str).str.strip() + ", " +
        df_event_info['area'].astype(str).str.strip() + ", " +
        df_event_info['country'].astype(str).str.strip()
    )

    ev_date = date.today().isoformat()
    df_event_info['start_date'] = ev_date
    df_event_info['end_date'] = ev_date

    if endpoint == 'daily':
        events = df_event_info[['index', 'location', 'start_date', 'end_date']].to_dict(orient='records')
    else:
        # expanding endpoint
        df_event_info['start_hour'] = 0
        df_event_info['end_hour'] = 23
        df_event_info['latitude'] = 0
        df_event_info['longitude'] = 0
        df_event_info['tag'] = df_event_info['location'].apply(lambda x: f"tag-{x[:20]}")
        events = df_event_info[['index', 'tag', 'location', 'start_date', 'end_date',
                                 'start_hour', 'end_hour', 'latitude', 'longitude']].to_dict(orient='records')

    return events


def build_summary_df(
    events: List[Dict[str, Any]],
    perils: List[str],
    elapsed: float
) -> pd.DataFrame:
    """Build summary DataFrame with timing and location/peril info."""

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

    for event in events:
        idx = event.get('index', 'N/A')
        location = event.get('location', 'N/A')
        start_date = event.get('start_date', 'N/A')
        end_date = event.get('end_date', 'N/A')
        rows.append({
            'metric': f'index_{idx}',
            'value': f"{location} ({start_date} to {end_date})"
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
    parser = argparse.ArgumentParser(description='Test BEV Prod API')
    parser.add_argument('--input-csv', type=str, help='CSV file with events to test', default=None)
    parser.add_argument('--locations-file', type=str, help='CSV of area/city mapping for random selection',
                        default='02_Data/area_city_location_id_mapping.csv')
    parser.add_argument('--num-locations', type=int, help='Number of random locations to sample', default=5)
    parser.add_argument('--perils', type=str, help='Comma-separated list of perils (prod format)', default='Rain')
    parser.add_argument('--window-days', type=int, help='window_days for expanding endpoint', default=0)
    parser.add_argument('--output-dir', type=str, help='Directory to save outputs', default='03_Analysis')
    parser.add_argument('--endpoint', type=str, choices=['daily', 'expanding'], default='daily')
    parser.add_argument('--verify-ssl', action='store_true', help='Verify SSL for prod')
    parser.add_argument('--ca-bundle', type=str, default=None, help='Path to CA bundle file')
    parser.add_argument('--prod-base', type=str, default=None, help='Override prod base URL')
    parser.add_argument('--random-seed', type=int, default=None, help='Random seed for location sampling')
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
        events = load_events_from_csv(args.input_csv, args.endpoint)
    else:
        logger.info(f"Sampling {args.num_locations} random locations")
        events = build_random_events(
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
    )

    # Save raw JSON
    json_path = output_dir / f"bev_prod_{ts}.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved JSON -> {json_path}")

    # Build summary
    summary_df = build_summary_df(events, perils, elapsed)

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
