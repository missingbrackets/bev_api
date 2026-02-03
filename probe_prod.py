import os
import requests

url = "https://prod-external-weather-api.birdseyeviewtechnologies.com/v1/in-depth/daily"
headers = {"x-api-key": os.environ.get('BEV_API_KEY_PROD') or os.environ.get('BEV_API_KEY'), "Content-Type": "application/json"}

payload = {"perils": ["Rain"], "events": [{"index": 0, "tag": "test", "location": "Dublin", "start_date": "2025-07-14", "end_date": "2025-07-14", "start_hour": 0, "end_hour": 23, "latitude": 53.3498, "longitude": -6.2603}]}


def try_mode(verify_flag):
    mode = 'verify=True' if verify_flag else 'verify=False'
    print(f'\n=== PRELIGHT ({mode}) for {url} ===')
    try:
        r = requests.options(url, headers=headers, verify=verify_flag, timeout=10)
        print('OPTIONS STATUS', r.status_code)
        print('OPTIONS ALLOW:', r.headers.get('Allow'))
        print('OPTIONS HEADERS:', dict(r.headers))
        print('OPTIONS BODY:', r.text[:1000])
    except requests.exceptions.SSLError as e:
        print('OPTIONS SSL ERR', repr(e))
    except Exception as e:
        print('OPTIONS ERR', repr(e))

    print('\nAttempting small POST')
    try:
        r = requests.post(url, headers=headers, json=payload, verify=verify_flag, timeout=20)
        print('POST STATUS', r.status_code)
        print('POST HEADERS:', dict(r.headers))
        print('POST BODY:', r.text[:2000])
    except requests.exceptions.SSLError as e:
        print('POST SSL ERR', repr(e))
    except Exception as e:
        print('POST ERR', repr(e))


if __name__ == '__main__':
    print('Using X-API-Key present:', bool(headers.get('x-api-key')))
    # Try with verification enabled first to surface certificate problems, then with verification disabled
    try_mode(True)
    try_mode(False)
