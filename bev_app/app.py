"""BEV Weather API Explorer – Streamlit App.

Run with:
    streamlit run app.py

Set your API key via environment variable BEV_API_KEY_PROD (or BEV_API_KEY),
or paste it into the sidebar.
"""

import os
import json
from datetime import date

import pandas as pd
import streamlit as st

from bev_client import call_bev_api, DEFAULT_PROD_BASE

ALL_PERILS = [
    "Rain",
    "MaxWindSpeed",
    "MaxWindGust",
    "Lightning",
    "Hail",
    "Tornado",
    "Snow",
    "SurfaceTemperature",
]

st.set_page_config(page_title="BEV Weather API", layout="wide")
st.title("BEV Weather API Explorer")

# ── Sidebar: API settings ──────────────────────────────────────────
with st.sidebar:
    st.header("API Settings")
    api_key = st.text_input(
        "API Key",
        value=os.environ.get("BEV_API_KEY_PROD", os.environ.get("BEV_API_KEY", "")),
        type="password",
    )
    base_url = st.text_input("Base URL", value=DEFAULT_PROD_BASE)
    endpoint = st.selectbox("Endpoint", ["daily", "expanding"])
    verify_ssl = st.checkbox("Verify SSL", value=False)

# ── Perils ──────────────────────────────────────────────────────────
st.subheader("Perils")
selected_perils = st.multiselect(
    "Select perils to query",
    ALL_PERILS,
    default=["Rain", "MaxWindSpeed", "MaxWindGust"],
)

# ── Locations ───────────────────────────────────────────────────────
st.subheader("Locations")

input_method = st.radio(
    "How do you want to provide locations?",
    ["Manual entry", "Upload CSV"],
    horizontal=True,
)

events = []
if input_method == "Manual entry":
    st.markdown(
        "Add one or more locations. Each needs **latitude**, **longitude**, "
        "**start date**, and **end date**."
    )

    if "locations" not in st.session_state:
        st.session_state.locations = [
            {"lat": 51.5074, "lon": -0.1278, "start": date.today(), "end": date.today()}
        ]

    def add_location():
        st.session_state.locations.append(
            {"lat": 0.0, "lon": 0.0, "start": date.today(), "end": date.today()}
        )

    def remove_location(i):
        st.session_state.locations.pop(i)

    for i, loc in enumerate(st.session_state.locations):
        cols = st.columns([2, 2, 2, 2, 1])
        with cols[0]:
            loc["lat"] = st.number_input(
                "Latitude", value=loc["lat"], format="%.4f", key=f"lat_{i}"
            )
        with cols[1]:
            loc["lon"] = st.number_input(
                "Longitude", value=loc["lon"], format="%.4f", key=f"lon_{i}"
            )
        with cols[2]:
            loc["start"] = st.date_input("Start date", value=loc["start"], key=f"sd_{i}")
        with cols[3]:
            loc["end"] = st.date_input("End date", value=loc["end"], key=f"ed_{i}")
        with cols[4]:
            st.write("")  # spacer
            if st.button("X", key=f"rm_{i}"):
                remove_location(i)
                st.rerun()

    st.button("+ Add location", on_click=add_location)

    for i, loc in enumerate(st.session_state.locations):
        events.append(
            {
                "index": i,
                "location": "",
                "latitude": loc["lat"],
                "longitude": loc["lon"],
                "start_date": loc["start"].isoformat(),
                "end_date": loc["end"].isoformat(),
            }
        )

else:
    st.markdown(
        "CSV must have columns: **index**, **latitude**, **longitude**, "
        "**start_date**, **end_date**. Optional: **location** (label only)."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_csv = pd.read_csv(uploaded)
        df_csv.columns = [c.strip().lower() for c in df_csv.columns]
        required = ["index", "start_date", "end_date", "latitude", "longitude"]
        missing = [c for c in required if c not in df_csv.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            df_csv["start_date"] = pd.to_datetime(df_csv["start_date"]).dt.strftime("%Y-%m-%d")
            df_csv["end_date"] = pd.to_datetime(df_csv["end_date"]).dt.strftime("%Y-%m-%d")
            df_csv["location"] = df_csv.get("location", "").fillna("")
            events = df_csv[
                ["index", "location", "start_date", "end_date", "latitude", "longitude"]
            ].to_dict(orient="records")
            st.success(f"Loaded {len(events)} events from CSV")
            st.dataframe(df_csv.head(10))

# ── Run ─────────────────────────────────────────────────────────────
st.divider()

if st.button("Run API Query", type="primary", disabled=not (api_key and events and selected_perils)):
    with st.spinner("Calling BEV API..."):
        try:
            results = call_bev_api(
                api_key=api_key,
                perils=selected_perils,
                events=events,
                endpoint=endpoint,
                base_url=base_url,
                verify_ssl=verify_ssl,
            )
            st.success(f"Got {len(results)} result records")

            # Flatten into DataFrame
            rows = []
            for r in results:
                idx = r.get("index")
                peril = r.get("peril")
                lat = r.get("latitude")
                lon = r.get("longitude")
                unit = r.get("unit")
                thresh_list = r.get("threshold") or []
                prob_list = r.get("probability") or []
                for t, p in zip(thresh_list, prob_list):
                    rows.append(
                        {
                            "index": idx,
                            "peril": peril,
                            "threshold": t,
                            "probability": p,
                            "latitude": lat,
                            "longitude": lon,
                            "unit": unit,
                        }
                    )

            if rows:
                df = pd.DataFrame(rows)
                st.subheader("Results")
                st.dataframe(df, use_container_width=True)

                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download CSV",
                        df.to_csv(index=False),
                        file_name="bev_results.csv",
                        mime="text/csv",
                    )
                with col2:
                    st.download_button(
                        "Download raw JSON",
                        json.dumps(results, indent=2),
                        file_name="bev_results.json",
                        mime="application/json",
                    )
            else:
                st.warning("API returned results but no threshold/probability data found")
                st.json(results)

        except Exception as exc:
            st.error(f"API call failed: {exc}")
            # Show response body if available
            if hasattr(exc, "response") and exc.response is not None:
                st.code(exc.response.text)

elif not api_key:
    st.warning("Enter your API key in the sidebar to get started.")
elif not events:
    st.info("Add at least one location above.")
elif not selected_perils:
    st.info("Select at least one peril.")
