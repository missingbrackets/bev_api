"""Run BEV API on staging and prod for the same dataset and compare results.

Usage:
    python compare_bev.py --events-file events.json --perils-file perils.json --output-dir 02_Data

Environment variables expected:
- BEV_API_KEY: staging API key (or default)
- BEV_API_KEY_PROD: prod API key (recommended)

Outputs (in output dir):
- bev_stage_<timestamp>.json
- bev_prod_<timestamp>.json
- bev_comparison_<timestamp>.xlsx (with sheets: stage, prod, comparison, summary)

"""
from pathlib import Path
import argparse
import json
import os
import time
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from bev_client import bev_task_batches_threaded

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_STAGE_BASE = 'https://nonprodstage-weather-api-wrapper.birdseyeviewtechnologies.com/model/'
# Prod API docs show endpoints at /v1/in-depth/<endpoint>
DEFAULT_PROD_BASE = 'https://prod-external-weather-api.birdseyeviewtechnologies.com/v1/in-depth'


def load_json_maybe(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(p.read_text())


def load_perils_for_envs(perils_file: str, endpoint: str):
    perils_df = pd.read_csv(perils_file)

    # Normalise column names (your CSV currently has a leading space in " peril_new")
    perils_df.columns = [c.strip() for c in perils_df.columns]

    # Filter to the endpoint you're running (daily/expanding)
    perils_df = perils_df.loc[perils_df["endpoint"] == endpoint].copy()

    # Stage uses the friendly peril names
    stage_perils = perils_df["Peril"].dropna().astype(str).str.strip().tolist()

    # Prod uses the enum-safe names from peril_new if present; else fall back to Peril
    if "peril_new" in perils_df.columns:
        prod_perils = perils_df["peril_new"].dropna().astype(str).str.strip().tolist()
    else:
        prod_perils = stage_perils

    if not stage_perils:
        stage_perils = ["Rain"]
    if not prod_perils:
        prod_perils = ["Rain"]

    return stage_perils, prod_perils


def map_stage_perils_to_prod(perils_file: str, endpoint: str, stage_perils: List[str]) -> List[str]:
    """
    Given a list of stage/friendly perils, return the prod enum-safe equivalents
    using perils_file mapping (Peril -> peril_new) filtered to the endpoint.
    """
    perils_df = pd.read_csv(perils_file)
    perils_df.columns = [c.strip() for c in perils_df.columns]
    perils_df = perils_df.loc[perils_df["endpoint"] == endpoint].copy()

    # Build mapping from friendly -> enum-safe
    if "peril_new" not in perils_df.columns:
        return stage_perils

    mapping = dict(
        zip(
            perils_df["Peril"].dropna().astype(str).str.strip(),
            perils_df["peril_new"].dropna().astype(str).str.strip(),
        )
    )

    # Map what we can; fall back to original if not found
    return [mapping.get(p.strip(), p.strip()) for p in stage_perils]


def flatten_results(results: List[Dict[str, Any]], env: str):
    """Convert API results into long DataFrame handling multiple possible response shapes.

    Supported shapes:
    - legacy: {index, peril, model: {threshold: value, ...}}
    - prod new: {index, peril, latitude, longitude, threshold: [...], probability: [...], unit}
    - expanding: includes window_index field
    """
    rows = []
    for r in results:
        idx = r.get('index')
        peril = r.get('peril')
        window_index = r.get('window_index')  # Present in expanding endpoint responses

        # Legacy response: 'model' is a dict mapping thresholds -> value
        if 'model' in r and isinstance(r.get('model'), dict):
            for thresh, val in r['model'].items():
                try:
                    valf = float(val)
                except Exception:
                    valf = None
                row = {
                    'index': idx,
                    'peril': peril,
                    'threshold': str(thresh),
                    'value': valf,
                    'env': env,
                }
                if window_index is not None:
                    row['window_index'] = window_index
                rows.append(row)
            continue

        # New prod response: parallel arrays 'threshold' and 'probability'
        if 'threshold' in r and 'probability' in r:
            thresh_list = r.get('threshold') or []
            prob_list = r.get('probability') or []
            for t, p in zip(thresh_list, prob_list):
                try:
                    pf = float(p)
                except Exception:
                    pf = None
                row = {
                    'index': idx,
                    'peril': peril,
                    'threshold': str(t),
                    'value': pf,
                    'env': env,
                }
                if window_index is not None:
                    row['window_index'] = window_index
                rows.append(row)
            continue

        # Fallback: if 'model' exists but not a dict, or unknown shape, attempt best-effort parsing
        model = r.get('model')
        if model:
            # try iterating items if possible
            try:
                for thresh, val in (model.items() if hasattr(model, 'items') else enumerate(model)):
                    row = {
                        'index': idx,
                        'peril': peril,
                        'threshold': str(thresh),
                        'value': float(val),
                        'env': env,
                    }
                    if window_index is not None:
                        row['window_index'] = window_index
                    rows.append(row)
            except Exception:
                # give up on this record
                continue

    return pd.DataFrame(rows)


def run_api(api_key: str, perils: List[str], endpoint: str, events: List[Dict[str, Any]],
            env_label: str, base_url: str, verify_ssl: bool, ca_bundle: str = None,
            window_days: int = 0) -> Tuple[pd.DataFrame, List[Dict[str, Any]], float]:
    """Run API and return (dataframe, raw_results, elapsed_seconds)."""
    start = time.time()
    endpoint_trailing = base_url.endswith('/')
    full_url = base_url.rstrip('/') + '/' + endpoint + ('/' if endpoint_trailing else '')
    logger.info(f"Running {env_label} ({base_url}) -> {full_url} for {len(events)} events and perils={perils}")

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
        logger.info(f"{env_label} call succeeded with base {base_url} (endpoint_trailing={endpoint_trailing})")
    except Exception as exc:
        logger.error(f"{env_label} call to {full_url} failed: {exc}")
        try:
            import requests
            if isinstance(exc, requests.exceptions.HTTPError) and hasattr(exc, 'response') and exc.response is not None:
                logger.error(f"Response status: {exc.response.status_code}")
                logger.error(f"Response body: {exc.response.text}")
        except Exception:
            pass
        raise

    elapsed = time.time() - start
    logger.info(f"{env_label} run finished in {elapsed:.2f}s; got {len(results)} records")

    df = flatten_results(results, env_label)
    return df, results, elapsed


def save_json(results: List[Dict[str, Any]], output_dir: Path, env_label: str, ts: int) -> Path:
    """Save raw JSON results."""
    json_path = output_dir / f"bev_{env_label}_{ts}.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved {env_label} JSON -> {json_path}")
    return json_path


def build_comparison_df(df_stage: pd.DataFrame, df_prod: pd.DataFrame,
                        stage_perils: List[str], prod_perils: List[str]) -> pd.DataFrame:
    """Build comparison DataFrame by joining on index, peril, window_index (if present), and closest threshold.

    Normalizes prod peril names to stage format before joining (e.g., CumulativeRain -> Cumulative Rain).
    For expanding endpoint, also joins on window_index.
    """

    # Convert thresholds to numeric for matching
    df_stage = df_stage.copy()
    df_prod = df_prod.copy()
    df_stage['threshold_num'] = pd.to_numeric(df_stage['threshold'], errors='coerce')
    df_prod['threshold_num'] = pd.to_numeric(df_prod['threshold'], errors='coerce')

    # Build reverse mapping: prod peril -> stage peril
    prod_to_stage_peril = dict(zip(prod_perils, stage_perils))

    # Normalize prod perils to stage format for joining
    df_prod['peril_normalized'] = df_prod['peril'].map(lambda p: prod_to_stage_peril.get(p, p))
    df_stage['peril_normalized'] = df_stage['peril']

    # Check if window_index is present (expanding endpoint)
    has_window_index = 'window_index' in df_stage.columns or 'window_index' in df_prod.columns

    # Ensure window_index column exists in both if either has it
    if has_window_index:
        if 'window_index' not in df_stage.columns:
            df_stage['window_index'] = 0
        if 'window_index' not in df_prod.columns:
            df_prod['window_index'] = 0

    comparison_rows = []

    # Build keys based on whether window_index is present
    if has_window_index:
        stage_keys = set(zip(df_stage['index'], df_stage['window_index'], df_stage['peril_normalized']))
        prod_keys = set(zip(df_prod['index'], df_prod['window_index'], df_prod['peril_normalized']))
        all_keys = stage_keys | prod_keys

        for idx, win_idx, peril_norm in all_keys:
            stage_subset = df_stage[
                (df_stage['index'] == idx) &
                (df_stage['window_index'] == win_idx) &
                (df_stage['peril_normalized'] == peril_norm)
            ]
            prod_subset = df_prod[
                (df_prod['index'] == idx) &
                (df_prod['window_index'] == win_idx) &
                (df_prod['peril_normalized'] == peril_norm)
            ]

            _process_comparison_subset(
                comparison_rows, stage_subset, prod_subset,
                idx, peril_norm, win_idx
            )
    else:
        stage_keys = set(zip(df_stage['index'], df_stage['peril_normalized']))
        prod_keys = set(zip(df_prod['index'], df_prod['peril_normalized']))
        all_keys = stage_keys | prod_keys

        for idx, peril_norm in all_keys:
            stage_subset = df_stage[
                (df_stage['index'] == idx) &
                (df_stage['peril_normalized'] == peril_norm)
            ]
            prod_subset = df_prod[
                (df_prod['index'] == idx) &
                (df_prod['peril_normalized'] == peril_norm)
            ]

            _process_comparison_subset(
                comparison_rows, stage_subset, prod_subset,
                idx, peril_norm, None
            )

    comparison_df = pd.DataFrame(comparison_rows)

    # Sort by index, window_index (if present), peril, threshold for readability
    if not comparison_df.empty:
        sort_cols = ['index']
        if has_window_index:
            sort_cols.append('window_index')
        sort_cols.extend(['peril_stage', 'threshold'])
        comparison_df = comparison_df.sort_values(sort_cols).reset_index(drop=True)

    return comparison_df


def _process_comparison_subset(
    comparison_rows: List[Dict],
    stage_subset: pd.DataFrame,
    prod_subset: pd.DataFrame,
    idx: int,
    peril_norm: str,
    win_idx: int = None
):
    """Process a single (index, [window_index], peril) subset for comparison."""

    # Get original peril names for output
    stage_peril = stage_subset['peril'].iloc[0] if not stage_subset.empty else peril_norm
    prod_peril = prod_subset['peril'].iloc[0] if not prod_subset.empty else peril_norm

    # Base row data
    def make_row(**kwargs):
        row = {'index': idx}
        if win_idx is not None:
            row['window_index'] = win_idx
        row.update(kwargs)
        return row

    if stage_subset.empty and prod_subset.empty:
        return

    if stage_subset.empty:
        # Only prod data exists
        for _, row in prod_subset.iterrows():
            comparison_rows.append(make_row(
                peril_stage=None,
                peril_prod=row['peril'],
                threshold=row['threshold'],
                value_stage=np.nan,
                value_prod=row['value'],
                abs_diff=np.nan,
            ))
        return

    if prod_subset.empty:
        # Only stage data exists
        for _, row in stage_subset.iterrows():
            comparison_rows.append(make_row(
                peril_stage=row['peril'],
                peril_prod=None,
                threshold=row['threshold'],
                value_stage=row['value'],
                value_prod=np.nan,
                abs_diff=np.nan,
            ))
        return

    # Both have data - match by closest threshold
    prod_thresholds = prod_subset['threshold_num'].dropna().values

    for _, stage_row in stage_subset.iterrows():
        stage_thresh = stage_row['threshold_num']
        stage_val = stage_row['value']

        if pd.isna(stage_thresh) or len(prod_thresholds) == 0:
            # Can't match numerically, try exact string match
            exact_match = prod_subset[prod_subset['threshold'] == stage_row['threshold']]
            if not exact_match.empty:
                prod_val = exact_match.iloc[0]['value']
                comparison_rows.append(make_row(
                    peril_stage=stage_peril,
                    peril_prod=prod_peril,
                    threshold=stage_row['threshold'],
                    value_stage=stage_val,
                    value_prod=prod_val,
                    abs_diff=abs(stage_val - prod_val) if pd.notna(stage_val) and pd.notna(prod_val) else np.nan,
                ))
            else:
                comparison_rows.append(make_row(
                    peril_stage=stage_peril,
                    peril_prod=prod_peril,
                    threshold=stage_row['threshold'],
                    value_stage=stage_val,
                    value_prod=np.nan,
                    abs_diff=np.nan,
                ))
            continue

        # Find closest prod threshold
        diffs = np.abs(prod_thresholds - stage_thresh)
        closest_idx = np.argmin(diffs)
        closest_prod_row = prod_subset[prod_subset['threshold_num'] == prod_thresholds[closest_idx]].iloc[0]
        prod_val = closest_prod_row['value']

        comparison_rows.append(make_row(
            peril_stage=stage_peril,
            peril_prod=prod_peril,
            threshold=stage_row['threshold'],
            closest_prod_threshold=closest_prod_row['threshold'],
            value_stage=stage_val,
            value_prod=prod_val,
            abs_diff=abs(stage_val - prod_val) if pd.notna(stage_val) and pd.notna(prod_val) else np.nan,
        ))


def build_summary_df(
    events: List[Dict[str, Any]],
    stage_perils: List[str],
    prod_perils: List[str],
    stage_elapsed: float,
    prod_elapsed: float
) -> pd.DataFrame:
    """Build summary DataFrame with timing and location/peril info."""

    # Timing section
    timing_rows = [
        {'metric': 'stage_elapsed_seconds', 'value': f"{stage_elapsed:.2f}"},
        {'metric': 'prod_elapsed_seconds', 'value': f"{prod_elapsed:.2f}"},
        {'metric': 'total_elapsed_seconds', 'value': f"{stage_elapsed + prod_elapsed:.2f}"},
        {'metric': '', 'value': ''},
        {'metric': 'stage_perils', 'value': ', '.join(stage_perils)},
        {'metric': 'prod_perils', 'value': ', '.join(prod_perils)},
        {'metric': 'num_perils', 'value': str(len(stage_perils))},
        {'metric': '', 'value': ''},
        {'metric': 'num_locations', 'value': str(len(events))},
        {'metric': '', 'value': ''},
        {'metric': '--- Locations ---', 'value': ''},
    ]

    # Add location mapping
    for event in events:
        idx = event.get('index', 'N/A')
        location = event.get('location', 'N/A')
        start_date = event.get('start_date', 'N/A')
        end_date = event.get('end_date', 'N/A')
        timing_rows.append({
            'metric': f'index_{idx}',
            'value': f"{location} ({start_date} to {end_date})"
        })

    return pd.DataFrame(timing_rows)


def write_excel_output(
    df_stage: pd.DataFrame,
    df_prod: pd.DataFrame,
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    ts: int
) -> Path:
    """Write all DataFrames to a single Excel file with multiple sheets."""

    excel_path = output_dir / f"bev_comparison_{ts}.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Stage response
        df_stage.to_excel(writer, sheet_name='stage', index=False)

        # Sheet 2: Prod response
        df_prod.to_excel(writer, sheet_name='prod', index=False)

        # Sheet 3: Comparison
        comparison_df.to_excel(writer, sheet_name='comparison', index=False)

        # Sheet 4: Summary (timing, locations, perils)
        summary_df.to_excel(writer, sheet_name='summary', index=False)

    logger.info(f"Saved Excel output -> {excel_path}")
    return excel_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perils-file', type=str, help='CSV file mapping endpoints to perils', default='02_Data/perils.csv')
    parser.add_argument('--events-file', type=str, help='JSON file with events list', default=None)
    parser.add_argument('--locations-file', type=str, help='CSV of area/city mapping', default='02_Data/area_city_location_id_mapping.csv')
    parser.add_argument('--num-locations', type=int, help='Number of random locations to sample from mapping', default=5)
    parser.add_argument('--perils', type=str, help='Comma-separated list of perils to test (overrides perils-file)', default=None)
    parser.add_argument('--window-days', type=int, help='window_days to pass for expanding/cumulative perils', default=0)
    parser.add_argument('--output-dir', type=str, help='Directory to save outputs', default='03_Analysis')
    parser.add_argument('--endpoint', type=str, choices=['daily', 'expanding'], default='daily')
    parser.add_argument('--verify-stage', action='store_true', help='Verify SSL for staging (default disabled)')
    parser.add_argument('--verify-prod', action='store_true', help='Verify SSL for prod (recommended)')
    parser.add_argument('--ca-bundle', type=str, default=None, help='Path to a CA bundle file to use for SSL verification (applies to both envs)')
    parser.add_argument('--stage-base', type=str, default=None, help='Override stage base URL')
    parser.add_argument('--prod-base', type=str, default=None, help='Override prod base URL')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load perils
    if args.perils:
        perils_list_stage = [p.strip() for p in args.perils.split(",") if p.strip()]
        perils_list_prod = map_stage_perils_to_prod(args.perils_file, args.endpoint, perils_list_stage)
    else:
        perils_list_stage, perils_list_prod = load_perils_for_envs(args.perils_file, args.endpoint)


    # Build events from mapping CSV or events-file
    # Note: expanding endpoint requires different formats for stage vs prod
    if args.events_file:
        events_raw = load_json_maybe(args.events_file)
        # For expanding endpoint, we need to build both formats from raw events
        if args.endpoint == 'expanding':
            # Stage format: index, location, start_date, end_date
            events_stage = [
                {k: v for k, v in e.items() if k in ['index', 'location', 'start_date', 'end_date']}
                for e in events_raw
            ]
            # Prod format: full event with all fields
            events_prod = []
            for e in events_raw:
                prod_event = {
                    'index': e.get('index', 0),
                    'tag': e.get('tag', f"tag-{str(e.get('location', ''))[:20]}"),
                    'location': e.get('location', ''),
                    'start_date': e.get('start_date', ''),
                    'end_date': e.get('end_date', ''),
                    'start_hour': e.get('start_hour', 0),
                    'end_hour': e.get('end_hour', 23),
                    'latitude': e.get('latitude', 0),
                    'longitude': e.get('longitude', 0),
                }
                events_prod.append(prod_event)
        else:
            events_stage = events_raw
            events_prod = events_raw
    else:
        # Read location mapping and sample random locations
        loc_df = pd.read_csv(args.locations_file, usecols=[0,1,2], names=['country','area','city'], header=0)
        loc_df = loc_df.dropna(subset=['country','area','city'])
        n = min(args.num_locations, len(loc_df))
        sampled = loc_df.sample(n=n, random_state=1)
        sampled = sampled.reset_index(drop=True)

        # Build event info DataFrame
        df_event_info = sampled.copy()
        df_event_info['index'] = df_event_info.index
        df_event_info['bev_location'] = (
            df_event_info['city'].astype(str).str.strip() + ", " +
            df_event_info['area'].astype(str).str.strip() + ", " +
            df_event_info['country'].astype(str).str.strip()
        )
        # Use provided event date (same start/end for simplicity)
        from datetime import date
        ev_date = date.today().isoformat()
        df_event_info['event_start_date'] = ev_date
        df_event_info['event_end_date'] = ev_date

        df_event_set = (
            df_event_info[["index", "bev_location", "event_start_date", "event_end_date"]]
            .rename(columns={
                "bev_location": "location",
                "event_start_date": "start_date",
                "event_end_date": "end_date",
            })
        )

        # For daily endpoint, both stage and prod use the same simple format
        if args.endpoint == 'daily':
            events_stage = df_event_set[["index", "location", "start_date", "end_date"]].to_dict(orient="records")
            events_prod = events_stage
        else:
            # expanding endpoint: stage uses simple format, prod uses full format
            # Stage format: index, location, start_date, end_date
            events_stage = df_event_set[["index", "location", "start_date", "end_date"]].to_dict(orient="records")

            # Prod format: include start_hour, end_hour, latitude, longitude and tag
            expanded = df_event_set.copy()
            expanded['start_hour'] = 0
            expanded['end_hour'] = 23
            expanded['latitude'] = 0
            expanded['longitude'] = 0
            expanded['tag'] = expanded['location'].apply(lambda x: f"tag-{x[:20]}")
            events_prod = (
                expanded[["index", "tag", "location", "start_date", "end_date", "start_hour", "end_hour", "latitude", "longitude"]]
                .to_dict(orient='records')
            )

    # API keys
    key_stage = os.environ.get('BEV_API_KEY')
    key_prod = os.environ.get('BEV_API_KEY_PROD') or os.environ.get('BEV_API_KEY')

    if not key_stage:
        raise RuntimeError('Staging API key not set in BEV_API_KEY')
    if not key_prod:
        raise RuntimeError('Prod API key not set in BEV_API_KEY_PROD or BEV_API_KEY')

    # Determine CA bundles and base URLs from CLI args (per-env overrides take precedence)
    ca_bundle = args.ca_bundle

    stage_base = args.stage_base or DEFAULT_STAGE_BASE
    prod_base = args.prod_base or DEFAULT_PROD_BASE

    # Log test configuration before running
    logger.info("=" * 60)
    logger.info("TEST CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Window days: {args.window_days}")
    logger.info("")
    logger.info("PERILS:")
    logger.info(f"  Stage: {perils_list_stage}")
    logger.info(f"  Prod:  {perils_list_prod}")
    logger.info("")
    logger.info(f"LOCATIONS ({len(events_stage)} total):")
    for event in events_stage:
        idx = event.get('index', 'N/A')
        location = event.get('location', 'N/A')
        start_date = event.get('start_date', 'N/A')
        end_date = event.get('end_date', 'N/A')
        logger.info(f"  [{idx}] {location} ({start_date} to {end_date})")
    logger.info("=" * 60)
    logger.info("")

    ts = int(time.time())

    # Run stage
    df_stage, results_stage, elapsed_stage = run_api(
        api_key=key_stage,
        perils=perils_list_stage,
        endpoint=args.endpoint,
        events=events_stage,
        env_label="stage",
        base_url=stage_base,
        verify_ssl=args.verify_stage,
        ca_bundle=ca_bundle,
        window_days=args.window_days,
    )
    save_json(results_stage, output_dir, "stage", ts)

    # Run prod
    df_prod, results_prod, elapsed_prod = run_api(
        api_key=key_prod,
        perils=perils_list_prod,
        endpoint=args.endpoint,
        events=events_prod,
        env_label="prod",
        base_url=prod_base,
        verify_ssl=args.verify_prod,
        ca_bundle=ca_bundle,
        window_days=args.window_days,
    )
    save_json(results_prod, output_dir, "prod", ts)

    # Build comparison DataFrame
    comparison_df = build_comparison_df(df_stage, df_prod, perils_list_stage, perils_list_prod)

    # Build summary DataFrame (use events_stage for location info - both have same locations)
    summary_df = build_summary_df(
        events=events_stage,
        stage_perils=perils_list_stage,
        prod_perils=perils_list_prod,
        stage_elapsed=elapsed_stage,
        prod_elapsed=elapsed_prod,
    )

    # Write Excel output
    excel_path = write_excel_output(df_stage, df_prod, comparison_df, summary_df, output_dir, ts)

    logger.info('Comparison complete')

    # Print quick summary
    if not comparison_df.empty:
        top_diffs = comparison_df.sort_values('abs_diff', ascending=False).head(10)
        logger.info('Top differences by abs_diff:')
        logger.info('\n' + top_diffs.to_string(index=False))
    else:
        logger.info('No comparison data available')


if __name__ == '__main__':
    main()
