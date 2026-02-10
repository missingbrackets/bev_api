"""CD rater task orchestration.

This module exposes the high-level task functions consumed by the CD rater
pipeline.  Currently the only task is adverse weather, but the structure
allows additional task sections to be added alongside it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from algorithms.tasks_section.adverse_weather import (
    load_location_mapping,
    lookup_locations,
    run_adverse_weather,
)

logger = logging.getLogger(__name__)


def run_adverse_weather_task(
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
    location_csv: Optional[str] = None,
    verify_ssl: bool = False,
    ca_bundle: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience wrapper: load locations then run adverse-weather API.

    This is the function the CD rater pipeline calls.  It:

    1. Loads the location mapping table (with lat/lon).
    2. Filters to the requested locations.
    3. Delegates to :func:`run_adverse_weather` which handles batching,
       retries, 500-error isolation, and probability normalisation.

    Parameters
    ----------
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
    location_csv : str, optional
        Override path to the location mapping CSV.
    verify_ssl : bool
        Whether to verify TLS certificates.
    ca_bundle : str, optional
        Path to a CA bundle file.

    Returns
    -------
    (daily_rows, expanding_rows, failed_events)
        See :func:`run_adverse_weather` for details.
    """
    mapping_df = load_location_mapping(csv_path=location_csv)
    logger.info("Loaded %d locations from mapping table.", len(mapping_df))

    locations_df = lookup_locations(
        mapping_df,
        location_ids=location_ids,
        countries=countries,
        areas=areas,
        cities=cities,
    )

    if locations_df.empty:
        logger.warning("No locations matched the given filters.")
        return [], [], []

    logger.info(
        "Filtered to %d locations for adverse-weather query.", len(locations_df),
    )

    return run_adverse_weather(
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
