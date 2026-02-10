"""CD rater task orchestration.

This module exposes the high-level task functions consumed by the CD rater
pipeline.  Currently the only task is adverse weather, but the structure
allows additional task sections to be added alongside it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import polars as pl

from algorithms.tasks_section.adverse_weather import (
    lookup_locations,
    run_adverse_weather,
)


def run_adverse_weather_task(
    hxd,
    location_ids: Optional[List[int]] = None,
    countries: Optional[List[str]] = None,
    areas: Optional[List[str]] = None,
    cities: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    perils_daily: Optional[List[str]] = None,
    perils_expanding: Optional[List[str]] = None,
    window_days: int = 0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    verify_ssl: bool = False,
    ca_bundle: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run adverse-weather task: load locations, call API, surface warnings.

    Parameters
    ----------
    hxd
        The HX runtime data object.  Used to read
        ``hxd.params.area_city_location_id_mapping_new`` and to write
        failed-event warnings into ``hxd.outputs.warnings``.
    location_ids, countries, areas, cities
        Optional filters passed to :func:`lookup_locations`.
    start_date, end_date : str, optional
        ISO date strings.  Default to today.
    perils_daily, perils_expanding : list[str], optional
        Override default peril lists.
    window_days : int
        Window days for the expanding endpoint.
    api_key : str, optional
        BEV API key (falls back to environment variables).
    base_url : str, optional
        API base URL (falls back to production default).
    verify_ssl : bool
        Whether to verify TLS certificates.
    ca_bundle : str, optional
        Path to a CA bundle file.

    Returns
    -------
    (daily_rows, expanding_rows, failed_events)
    """
    # Load the location mapping from the framework parameter table.
    event_location_mapping = pl.from_pandas(
        hxd.params.area_city_location_id_mapping_new
    )
    mapping_df = event_location_mapping.to_pandas()

    locations_df = lookup_locations(
        mapping_df,
        location_ids=location_ids,
        countries=countries,
        areas=areas,
        cities=cities,
    )

    if locations_df.empty:
        return [], [], []

    daily_rows, expanding_rows, failed_events, warnings = run_adverse_weather(
        locations_df=locations_df,
        start_date=start_date,
        end_date=end_date,
        perils_daily=perils_daily,
        perils_expanding=perils_expanding,
        window_days=window_days,
        api_key=api_key,
        base_url=base_url,
        verify_ssl=verify_ssl,
        ca_bundle=ca_bundle,
    )

    # Surface failed-event warnings via the framework validation pattern.
    if warnings:
        df_params = hxd.params.validation.copy()
        df_warnings = df_params[df_params["type"] == "warning"]
        df_warnings = pd.concat(
            [df_warnings, pd.DataFrame(warnings)],
            ignore_index=True,
        )
        hxd.outputs.warnings = df_warnings[["label", "message"]].to_dict(
            orient="records"
        )

    return daily_rows, expanding_rows, failed_events
