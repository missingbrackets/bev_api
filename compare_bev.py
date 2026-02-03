"""Run BEV API on staging and prod for the same dataset and compare results.

Usage:
    python compare_bev.py --events-file events.json --perils-file perils.json --output-dir 02_Data

Environment variables expected:
- BEV_API_KEY: staging API key (or default)
- BEV_API_KEY_PROD: prod API key (recommended)

Outputs (in output dir):
- bev_stage_<timestamp>.json / .csv
- bev_prod_<timestamp>.json / .csv
- bev_comparison_<timestamp>.csv

"""
from pathlib import Path
import argparse
import json
import os
import time
import logging
from typing import List, Dict, Any

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
    """
    rows = []
    for r in results:
        idx = r.get('index')
        peril = r.get('peril')

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
                    'value': valf,
                    'env': env,
                })
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
                rows.append({
                    'index': idx,
                    'peril': peril,
                    'threshold': str(t),
                    'value': pf,
                    'env': env,
                })
            continue

        # Fallback: if 'model' exists but not a dict, or unknown shape, attempt best-effort parsing
        model = r.get('model')
        if model:
            # try iterating items if possible
            try:
                for thresh, val in (model.items() if hasattr(model, 'items') else enumerate(model)):
                    rows.append({
                        'index': idx,
                        'peril': peril,
                        'threshold': str(thresh),
                        'value': float(val),
                        'env': env,
                    })
            except Exception:
                # give up on this record
                continue

    return pd.DataFrame(rows)


def run_and_save(api_key: str, perils: List[str], endpoint: str, events: List[Dict[str, Any]], env_label: str, output_dir: Path, base_url: str, verify_ssl: bool, ca_bundle: str = None, window_days: int = 0):
    start = time.time()
    # Simpler behavior: use the base_url as the canonical base and determine whether
    # to include a trailing slash from whether base_url itself ends with '/'
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

    ts = int(time.time())
    json_path = output_dir / f"bev_{env_label}_{ts}.json"
    csv_path = output_dir / f"bev_{env_label}_{ts}.csv"

    json_path.write_text(json.dumps(results, indent=2))

    df = flatten_results(results, env_label)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved {env_label} JSON -> {json_path} and CSV -> {csv_path}")
    return df, json_path, csv_path


import numpy as np


def compare_dfs(df_stage: pd.DataFrame, df_prod: pd.DataFrame, output_dir: Path):
    # merge on index, peril, threshold
    left = df_stage.rename(columns={'value': 'value_stage'})
    right = df_prod.rename(columns={'value': 'value_prod'})
    merged = left.merge(right, on=['index', 'peril', 'threshold'], how='outer')

    # normalize types and fill missing with NaN to allow meaningful percentages
    merged['value_stage'] = pd.to_numeric(merged['value_stage'], errors='coerce')
    merged['value_prod'] = pd.to_numeric(merged['value_prod'], errors='coerce')

    merged['abs_diff'] = (merged['value_stage'] - merged['value_prod']).abs()

    # pct_diff: relative to prod when available, otherwise relative to stage; if both zero -> NaN
    def _pct_diff(row):
        a = row['value_stage']
        b = row['value_prod']
        if pd.isna(a) and pd.isna(b):
            return np.nan
        if b not in (0, 0.0, None) and not pd.isna(b):
            return (a - b) / b * 100.0
        if a not in (0, 0.0, None) and not pd.isna(a):
            return (a - b) / a * 100.0
        return np.nan

    merged['pct_diff'] = merged.apply(_pct_diff, axis=1)

    ts = int(time.time())
    out_csv_long = output_dir / f"bev_comparison_long_{ts}.csv"
    merged.to_csv(out_csv_long, index=False)
    logger.info(f"Saved long-form comparison CSV -> {out_csv_long}")

    # Pivot to wide format: one row per (index, peril), columns for stage/prod thresholds side-by-side
    pivot = merged.pivot_table(index=['index', 'peril'], columns='threshold', values=['value_stage', 'value_prod'], aggfunc='first')
    # Flatten multiindex columns: value_stage_0, value_prod_0, etc.
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
    pivot = pivot.reset_index()
    out_csv_wide = output_dir / f"bev_comparison_wide_{ts}.csv"
    pivot.to_csv(out_csv_wide, index=False)
    logger.info(f"Saved wide-form comparison CSV -> {out_csv_wide}")

    # Summary metrics per (index, peril)
    def summarize(g: pd.DataFrame):
        a = g['value_stage'].to_numpy(dtype=float)
        b = g['value_prod'].to_numpy(dtype=float)
        mask = ~(np.isnan(a) & np.isnan(b))
        a = a[mask]
        b = b[mask]
        if len(a) == 0:
            return pd.Series({'mae': np.nan, 'rmse': np.nan, 'max_abs_diff': np.nan, 'mean_pct_diff': np.nan, 'corr': np.nan, 'count': 0})
        mae = np.nanmean(np.abs(a - b))
        rmse = np.sqrt(np.nanmean((a - b) ** 2))
        max_abs = np.nanmax(np.abs(a - b))
        # percent differences relative to prod where possible
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = np.abs((a - b) / np.where(b != 0, b, np.nan)) * 100.0
            mean_pct = np.nanmean(pct)
        corr = np.nan
        if len(a) > 1 and np.nanstd(a) > 0 and np.nanstd(b) > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
        return pd.Series({'mae': mae, 'rmse': rmse, 'max_abs_diff': max_abs, 'mean_pct_diff': mean_pct, 'corr': corr, 'count': len(a)})

    summary = merged.groupby(['index', 'peril']).apply(summarize).reset_index()
    out_csv_summary = output_dir / f"bev_comparison_summary_{ts}.csv"
    summary.to_csv(out_csv_summary, index=False)
    logger.info(f"Saved summary CSV -> {out_csv_summary}")

    return merged, out_csv_long, out_csv_wide, out_csv_summary


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
    if args.events_file:
        events = load_json_maybe(args.events_file)
    else:
        # Read location mapping and sample random locations
        loc_df = pd.read_csv(args.locations_file, usecols=[0,1,2], names=['country','area','city'], header=0)
        loc_df = loc_df.dropna(subset=['country','area','city'])
        n = min(args.num_locations, len(loc_df))
        sampled = loc_df.sample(n=n, random_state=1)
        sampled = sampled.reset_index(drop=True)

        # Build event info DataFrame similar to your snippet
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
        df_event_info['requires_cumulative'] = False

        df_event_set = (
            df_event_info[["index", "bev_location", "event_start_date", "event_end_date", "requires_cumulative"]]
            .assign(**{
                col: df_event_info[col].astype(str)
                for col in df_event_info.select_dtypes(include=['datetime', 'datetimetz']).columns
            })
            .rename(columns={
                "bev_location": "location",
                "event_start_date": "start_date",
                "event_end_date": "end_date",
            })
        )

        # For daily endpoint use simple event shape; for expanded/prod include hours and lat/lon defaults
        if args.endpoint == 'daily':
            daily_event_set = (
                df_event_set[["index", "location", "start_date", "end_date"]]
                .to_dict(orient="records")
            )
            events = daily_event_set
        else:
            # expanding endpoint: include start_hour, end_hour, latitude, longitude and tag
            expanded = df_event_set.copy()
            expanded['start_hour'] = 0
            expanded['end_hour'] = 23
            expanded['latitude'] = 0
            expanded['longitude'] = 0
            expanded['tag'] = expanded['location'].apply(lambda x: f"tag-{x[:20]}")
            events = (
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
    # Use single --ca-bundle for both envs (keeps CLI simple). Users can override base URLs explicitly.
    ca_bundle = args.ca_bundle

    stage_base = args.stage_base or DEFAULT_STAGE_BASE
    prod_base = args.prod_base or DEFAULT_PROD_BASE

    # Run stage
    df_stage, js_stage, csv_stage = run_and_save(
        api_key=key_stage,
        perils=perils_list_stage,
        endpoint=args.endpoint,
        events=events,
        env_label="stage",
        output_dir=output_dir,
        base_url=stage_base,
        verify_ssl=args.verify_stage,
        ca_bundle=ca_bundle,
        window_days=args.window_days,
    )

    # Run prod
    df_prod, js_prod, csv_prod = run_and_save(
        api_key=key_prod,
        perils=perils_list_prod,
        endpoint=args.endpoint,
        events=events,
        env_label="prod",
        output_dir=output_dir,
        base_url=prod_base,
        verify_ssl=args.verify_prod,
        ca_bundle=ca_bundle,
        window_days=args.window_days,
    )

    # Compare
    merged, cmp_csv_long, cmp_csv_wide, cmp_csv_summary = compare_dfs(df_stage, df_prod, output_dir)
    logger.info('Comparison complete')

    # Print quick summary from the summary CSV
    try:
        summary = pd.read_csv(cmp_csv_summary)
        top_summary = summary.sort_values('rmse', ascending=False).head(10)
        logger.info('Top RMSE per (index, peril):')
        logger.info('\n' + top_summary.to_string(index=False))
    except Exception:
        # Fall back to previous long-form view
        top_diffs = merged.sort_values('abs_diff', ascending=False).head(10)
        logger.info('Top differences (long):')
        logger.info('\n' + top_diffs.to_string(index=False))


if __name__ == '__main__':
    main()
