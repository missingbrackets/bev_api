import os
import requests

base = 'https://nonprodstage-weather-api-wrapper.birdseyeviewtechnologies.com/model'
for suffix in ['', '/']:
    url = base + '/daily' + suffix
    headers = {'x-api-key': os.environ.get('BEV_API_KEY') or os.environ.get('BEV_API_KEY_PROD'), 'Content-Type': 'application/json'}
    print('\nURL:', url)
    try:
        r = requests.options(url, headers=headers, verify=False, timeout=10)
        print('OPTIONS', r.status_code, 'Allow:', r.headers.get('Allow'))
        print('OPTIONS body:', r.text)
    except Exception as e:
        print('OPTIONS ERR', e)
    payload = {"perils":["Rain"],"events":[{"index":0,"tag":"test","location":"Dublin","start_date":"2025-07-14","end_date":"2025-07-14","start_hour":0,"end_hour":23,"latitude":53.3498,"longitude":-6.2603}]}
    try:
        r2 = requests.post(url, headers=headers, json=payload, verify=False, timeout=20)
        print('POST', r2.status_code)
        print('POST body:', r2.text[:500])
    except Exception as e:
        print('POST ERR', e)
