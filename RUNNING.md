BEV Compare Tool - Quick Runbook

Prerequisites
- Python 3.10+ in a virtual environment (this project uses a .venv)
- Install dependencies: pip install -r requirements.txt  (or pip install requests pandas pytest)

Environment
- Set `BEV_API_KEY` for staging calls.
- Set `BEV_API_KEY_PROD` for production calls (optional; falls back to `BEV_API_KEY`).

Run unit tests
- Activate the venv (PowerShell): .\.venv\Scripts\Activate.ps1 (or source .venv/bin/activate on macOS/Linux)
- Run: python -m pytest -q

Run compare (stage vs prod)
- By default the script will sample a small set of locations and run both envs; outputs are written to `02_Data`.

Minimal (insecure TLS):
```
BEV_API_KEY_PROD=<your_prod_key> python compare_bev.py --output-dir 02_Data
```

Secure (recommended if you have a CA bundle):
```
BEV_API_KEY_PROD=<your_prod_key> python compare_bev.py --output-dir 02_Data --prod-base https://prod-external-weather-api.birdseyeviewtechnologies.com/v1/in-depth --ca-bundle /path/to/ca.pem --verify-prod
```

Useful CLI flags
- --locations-file: CSV mapping file (default 02_Data/area_city_location_id_mapping.csv)
- --num-locations: number of random locations to sample (default 5)
- --perils-file: CSV file mapping endpoints to perils (default 02_Data/perils.csv)
- --perils: comma-separated list of perils to test (overrides --perils-file)
- --endpoint: daily or expanding (default daily)
- --window-days: integer passed to API for cumulative/expanding calls (default 0)

Notes
- If the prod server TLS chain is incomplete, add --ca-bundle /path/to/ca.pem or ask BEV to fix the certificate chain.
- If you want a specific events set, pass --events-file my_events.json and --perils or --perils-file to control perils.

If you want, run `python compare_bev.py --help` for full usage details.
